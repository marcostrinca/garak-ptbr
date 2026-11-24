# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "company_run",
    Path(__file__).parents[1] / "tools" / "company_run" / "run.py",
)
company_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(company_run)


def _company(tmp_path):
    return {
        "name": "shop",
        "target": {
            "uri": "http://real.example/chat",
            "preflight_command": "check-test-mode",
            "stop_patterns": ["real-cart"],
            "stop_file": str(tmp_path / "STOP"),
        },
        "agent": {"tools": [{"name": "a"}, {"name": "b"}, {"name": "c"}]},
        # an override that would point at the real target again
        "garak": {
            "plugins": {
                "target_type": "function",
                "target_name": "mod#real",
                "generators": {"rest": {"RestGenerator": {"uri": "http://real.example/chat"}}},
            }
        },
    }


def test_rehearsal_never_uses_the_real_target(tmp_path):
    config = company_run.build_config(
        _company(tmp_path), "goat", "pt", rehearsal_uri="http://127.0.0.1:1/"
    )
    plugins = config["plugins"]
    target = plugins["generators"]["rest"]["RestGenerator"]
    assert plugins["target_type"] == "rest" and "target_name" not in plugins
    assert list(plugins["generators"]) == ["rest"]
    assert target["uri"] == "http://127.0.0.1:1/"
    assert "preflight_command" not in target
    assert target["stop_file"] != str(tmp_path / "STOP")


def test_live_run_stops_before_next_family_when_stop_file_appears(tmp_path):
    company = _company(tmp_path)
    started = []

    def run_one(family, lang):
        started.append((family, lang))
        (tmp_path / "STOP").write_text("matched")
        return 1

    done, finished = company_run.run_all(
        company, [("lmrc", "pt"), ("lmrc", "en")], run_one
    )
    assert started == [("lmrc", "pt")]
    assert not finished


def test_topic_blocked_topics_only_in_their_language(tmp_path):
    company = _company(tmp_path)
    company["agent"]["blocked_topics"] = {"pt": ["arma"]}
    pt = company_run.build_config(company, "topic", "pt")
    assert pt["plugins"]["probes"]["topic"]["WordnetBlockedWords"]["target_topics"] == ["arma"]
    assert "topic.WordnetBlockedWords" in company_run.probe_spec(company, "topic", "pt")
    en = company_run.build_config(company, "topic", "en")
    assert "topic" not in en["plugins"]["probes"]
    assert (
        company_run.probe_spec(company, "topic", "en") == "topic.WordnetControversial"
    )


def test_web_injection_exfil_domain_never_resolves(tmp_path):
    company = _company(tmp_path)
    company["agent"]["image_format"] = "[IMG]{uri}"
    config = company_run.build_config(company, "web_injection", "pt")
    image_exfil = config["plugins"]["probes"]["web_injection"]["ImageExfil"]
    assert image_exfil["image_format"] == "[IMG]{uri}"
    assert image_exfil["domains"]
    assert all(d.endswith(".invalid") for d in image_exfil["domains"])
