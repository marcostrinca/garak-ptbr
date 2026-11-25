#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run garak against one company's agent from a single config file.

    python tools/company_run/run.py company.yaml                 show the plan, send nothing
    python tools/company_run/run.py company.yaml --rehearsal     fake local target, reduced volume
    python tools/company_run/run.py company.yaml --live          the real target in company.yaml

Narrow the run with --family NAME and --lang pt|en (both repeatable).
Each family and language is one garak run. A live run stops before the next
one when the target's stop_file exists. See README.md next to this file.
"""

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import yaml

# family -> (probes, languages it runs in by default)
FAMILIES = {
    "dan": ("dan", ["en"]),
    "lmrc": ("lmrc", ["pt", "en"]),
    "misleading": ("misleading", ["pt", "en"]),
    "promptinject": ("promptinject", ["pt", "en"]),
    "sysprompt_extraction": ("sysprompt_extraction", ["pt", "en"]),
    "donotanswer": (
        "donotanswer.DiscriminationExclusionToxicityHatefulOffensive,"
        "donotanswer.HumanChatbox,donotanswer.InformationHazard,"
        "donotanswer.MaliciousUses,donotanswer.MisinformationHarms",
        ["pt", "en"],
    ),
    "goat": ("goat.GOATAttack", ["pt", "en"]),
    "fitd": ("fitd.FITD", ["pt", "en"]),
    "agent_breaker": ("agent_breaker.AgentBreaker", ["pt", "en"]),
    "topic": ("topic.WordnetControversial", ["pt", "en"]),
    "grandma": ("grandma.Substances", ["pt", "en"]),
    "doctor": ("doctor.Puppetry,doctor.Bypass,doctor.BypassLeet", ["pt", "en"]),
    "phrasing": ("phrasing.PastTense,phrasing.FutureTense", ["pt", "en"]),
    "web_injection": ("web_injection.ImageExfil", ["pt", "en"]),
}

# web_injection: .invalid never resolves (RFC 2606), so a leaked URI reaches no one
EXFIL_DOMAINS = ["images-cdn.invalid"]


def probe_spec(company, family, lang):
    """--probes for one family and language"""
    spec = FAMILIES[family][0]
    if family == "topic" and company.get("agent", {}).get("blocked_topics", {}).get(lang):
        spec += ",topic.WordnetBlockedWords"
    return spec

DEFAULT_MODEL = {
    "type": "openai.OpenAICompatible",
    "name": "deepseek-ai/DeepSeek-V4.1-Flash",
    "uri": "https://api.together.xyz/v1",
}

# rehearsal: a few prompts per probe, only to see that everything loads and the judges answer
REHEARSAL = {
    "run": {"soft_probe_prompt_cap": 2},
    "plugins": {
        "probes": {
            "goat": {"GOATAttack": {"num_goals": 1, "max_calls_per_conv": 2}},
            "fitd": {"FITD": {"initial_prompt_count": 1, "max_calls_per_conv": 3}},
            "agent_breaker": {"AgentBreaker": {"max_attempts_per_tool": 1}},
        }
    },
}
REHEARSAL_TOOLS = 2
REHEARSAL_REPLY = (
    "Desculpe, não posso ajudar com isso. Posso ajudar com outra coisa da loja?"
)


def merge(a, b):
    a = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            a[k] = merge(a[k], v)
        else:
            a[k] = copy.deepcopy(v)
    return a


def plan(company, families=None, langs=None):
    """(family, lang) pairs to run, in order"""
    chosen = company.get("run", {}).get("families") or {
        f: v[1] for f, v in FAMILIES.items()
    }
    pairs = []
    for family, family_langs in chosen.items():
        if family not in FAMILIES:
            raise ValueError(f"unknown family {family!r}; known: {', '.join(FAMILIES)}")
        if families and family not in families:
            continue
        if family == "sysprompt_extraction" and not company.get("agent", {}).get(
            "system_prompt_file"
        ):
            continue  # without the real prompt it would send fake system prompts
        for lang in family_langs:
            if langs and lang not in langs:
                continue
            pairs.append((family, lang))
    return pairs


def build_config(company, family, lang, rehearsal_uri=None):
    """garak config for one family and language; rehearsal_uri replaces the real target"""
    model = DEFAULT_MODEL | company.get("model", {})
    attacker = {
        "red_team_model_type": model["type"],
        "red_team_model_name": model["name"],
        "red_team_model_config": {
            "uri": model["uri"],
            # reasoning models count their thinking in max_tokens; at 8192 the
            # agent_breaker tool analysis came back cut or empty
            "max_tokens": 32768,
            "temperature": 0.7,
            "suppressed_params": ["stop"],
        },
    }
    agent = company.get("agent", {})
    run = company.get("run", {})

    target = copy.deepcopy(company["target"])
    if rehearsal_uri:
        # never the real target: no preflight command, a throwaway stop file
        target = {
            "uri": rehearsal_uri,
            "req_template_json_object": {"message": "$INPUT", "session": "$SESSION"},
            "response_json": True,
            "response_json_field": "$.text",
            "session_json_field": "$.session",
            "stop_patterns": target.get("stop_patterns", []),
            "stop_file": os.path.join(tempfile.mkdtemp(), "stop"),
        }

    probes = {
        "goat": {"GOATAttack": attacker | {"num_goals": -1, "max_calls_per_conv": 5}},
        "fitd": {
            "FITD": attacker
            | {"initial_prompt_count": 5, "num_level_queries": 3, "max_calls_per_conv": 10}
        },
        "agent_breaker": {
            "AgentBreaker": attacker
            | {
                "max_attempts_per_tool": 5,
                "agent_purpose": agent.get("purpose", ""),
                "tools": agent.get("tools", []),
            }
        },
    }
    if agent.get("system_prompt_file"):
        probes["sysprompt_extraction"] = {
            "SystemPromptExtraction": {"system_prompt_file": agent["system_prompt_file"]}
        }
    probes["web_injection"] = {"ImageExfil": {"domains": EXFIL_DOMAINS}}
    if agent.get("image_format"):
        probes["web_injection"]["ImageExfil"]["image_format"] = agent["image_format"]
    blocked = agent.get("blocked_topics", {}).get(lang)
    if blocked:
        probes["topic"] = {"WordnetBlockedWords": {"target_topics": blocked}}
    goals = run.get("goat_goals_file", {}).get(lang)
    if goals:
        probes["goat"]["GOATAttack"]["goals_file"] = goals

    config = {
        "run": {"generations": 1, "seed": run.get("seed", 7), "target_lang": lang},
        "system": {"parallel_attempts": False, "lite": True},
        "plugins": {
            "target_type": "rest",
            "generators": {"rest": {"RestGenerator": target}},
            "probes": probes,
            "detectors": {
                "ptbr": {
                    "detector_model_type": model["type"],
                    "detector_model_name": model["name"],
                    "detector_model_config": {
                        "uri": model["uri"],
                        "temperature": 0,
                        "max_tokens": 4000,
                    },
                    "target_description": agent.get("description", "um assistente"),
                },
                "agent_breaker": {
                    "AgentBreakerResult": {
                        "detector_model_type": model["type"],
                        "detector_model_name": model["name"],
                        "detector_model_config": {
                            "uri": model["uri"],
                            "max_tokens": 8192,
                            "temperature": 0,
                            "suppressed_params": ["n", "stop"],
                        },
                    }
                },
            },
        },
        "reporting": {
            "report_prefix": f"{company['name']}-{family.replace('_', '')}-{lang}"
            + ("-rehearsal" if rehearsal_uri else "")
        },
    }
    if run.get("report_dir"):
        config["reporting"]["report_dir"] = run["report_dir"]
    agent_lang = agent.get("language", "pt")
    if lang == "pt":
        # a pt run needs a translator loaded; the families here send reviewed pt text
        translator = {
            "model_type": "llm.LLMTranslator",
            "translation_model_type": model["type"],
            "translation_model_name": model["name"],
            "translation_model_config": {
                "uri": model["uri"],
                "max_tokens": 4000,
                "temperature": 0.1,
                "suppressed_params": ["stop"],
            },
        }
        config["run"]["langproviders"] = [
            {"language": "en,pt"} | translator,
            {"language": "pt,en"} | translator,
        ]
    elif agent_lang != lang:
        config["run"]["response_lang"] = agent_lang
    config = merge(config, company.get("garak", {}))
    if rehearsal_uri:
        config = merge(config, REHEARSAL)
        ab = config["plugins"]["probes"]["agent_breaker"]["AgentBreaker"]
        ab["tools"] = ab["tools"][:REHEARSAL_TOOLS]
        # garak overrides in the company file must not bring the real target back
        config["plugins"]["target_type"] = "rest"
        config["plugins"].pop("target_name", None)
        config["plugins"]["generators"] = {"rest": {"RestGenerator": target}}
    return config


def stop_file(company):
    return company["target"].get("stop_file")


def run_all(company, pairs, run_one):
    """Run each pair with run_one(family, lang); stop when the stop file shows up"""
    done = []
    for family, lang in pairs:
        stop = stop_file(company)
        if stop and os.path.exists(stop):
            print(f"STOP: {stop} exists, not starting {family} {lang}")
            return done, False
        done.append((family, lang, run_one(family, lang)))
    stop = stop_file(company)
    return done, not (stop and os.path.exists(stop))


def report_path(company, config):
    report_dir = company.get("run", {}).get("report_dir") or "garak_runs"
    if not os.path.isabs(report_dir):
        data_home = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
        report_dir = os.path.join(data_home, "garak", report_dir)
    return os.path.join(report_dir, config["reporting"]["report_prefix"] + ".report.jsonl")


def summary(report_path):
    """Detector results and attempts per probe, from a garak report"""
    lines, attempts = [], {}
    with open(report_path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("entry_type") == "eval":
                lines.append(
                    f"  {entry['probe']} > {entry['detector']}: passed {entry['passed']}"
                    f" failed {entry['fails']} no score {entry.get('nones')}"
                    f" of {entry['total_evaluated']}"
                )
            elif entry.get("entry_type") == "attempt" and entry.get("status") == 2:
                attempts[entry["probe_classname"]] = (
                    attempts.get(entry["probe_classname"], 0) + 1
                )
    return [f"  attempts {attempts}"] + lines


class _FakeTarget(BaseHTTPRequestHandler):
    sent = 0

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).sent += 1
        session = body.get("session") or f"s{type(self).sent}"
        reply = json.dumps({"text": REHEARSAL_REPLY, "session": session}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *args):
        pass


def load_env_file(path):
    env = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip("'\"")
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("company_file")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--rehearsal", action="store_true")
    mode.add_argument("--live", action="store_true")
    parser.add_argument("--family", action="append")
    parser.add_argument("--lang", action="append")
    args = parser.parse_args(argv)

    with open(args.company_file, encoding="utf-8") as f:
        company = yaml.safe_load(f)
    pairs = plan(company, args.family, args.lang)

    if not (args.rehearsal or args.live):
        print(f"{company['name']}: {len(pairs)} garak runs, nothing sent")
        if not company.get("agent", {}).get("system_prompt_file"):
            print("  sysprompt_extraction skipped: agent.system_prompt_file not set")
        for family, lang in pairs:
            print(f"  {family} {lang}: {probe_spec(company, family, lang)}")
        print("use --rehearsal (fake target) or --live (real target)")
        return 0

    env = dict(os.environ)
    if company.get("env_file"):
        env |= load_env_file(company["env_file"])
    model = DEFAULT_MODEL | company.get("model", {})
    if model["type"] == "openai.OpenAICompatible" and not env.get(
        "OPENAICOMPATIBLE_API_KEY"
    ):
        print("OPENAICOMPATIBLE_API_KEY is not set (environment or env_file)")
        return 2

    server = None
    if args.rehearsal:
        server = HTTPServer(("127.0.0.1", 0), _FakeTarget)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        rehearsal_uri = f"http://127.0.0.1:{server.server_port}/"
        # the fake target is local: keep it off any HTTP proxy
        for var in ("NO_PROXY", "no_proxy"):
            env[var] = ",".join(x for x in (env.get(var), "127.0.0.1") if x)
    else:
        rehearsal_uri = None

    workdir = tempfile.mkdtemp(prefix="company_run_")

    def run_one(family, lang):
        config = build_config(company, family, lang, rehearsal_uri)
        path = os.path.join(workdir, config["reporting"]["report_prefix"] + ".yaml")
        with open(os.open(path, os.O_WRONLY | os.O_CREAT, 0o600), "w") as f:
            yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
        sent_before = _FakeTarget.sent
        print(f"== {family} {lang}", flush=True)
        code = subprocess.call(
            [sys.executable, "-m", "garak", "--config", path, "--probes", probe_spec(company, family, lang)],
            env=env,
        )
        if server:
            print(f"  {_FakeTarget.sent - sent_before} messages to the fake target")
        report = report_path(company, config)
        if os.path.exists(report):
            print("\n".join(summary(report)))
        return code

    done, finished = run_all(company, pairs, run_one)
    if server:
        server.shutdown()
    print(f"\n{company['name']}: {len(done)} of {len(pairs)} runs")
    for family, lang, code in done:
        print(f"{family} {lang}: exit {code}")
    if not finished:
        print(f"STOPPED: {stop_file(company)} exists; look at it before removing it")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
