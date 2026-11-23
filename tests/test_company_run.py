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
