# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probes that translate their payloads before transforming them (Probe._localize)"""

import base64
import re
import tempfile

import pytest

import garak.generators.base
import garak.payloads
import garak.probes.base
import garak.probes.encoding
import garak.resources.encodings
from garak import _config, _plugins
from garak.attempt import Conversation, Message, Turn


class FakeTranslator:
    """Deterministic stand-in for a language provider; marks text with an accented suffix"""

    def __init__(self, source_lang, target_lang):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.calls = []

    def get_text(self, prompts, notify_callback=None):
        self.calls.append(list(prompts))
        if notify_callback:
            notify_callback(len(prompts))
        if self.source_lang == self.target_lang:
            return list(prompts)
        return [f"{p} em português, ação" for p in prompts]


@pytest.fixture
def fake_pt(monkeypatch):
    """Run probes as if translating en -> pt, with no model or network"""
    _config.load_base_config()
    _config.run.seed = 42
    report = tempfile.NamedTemporaryFile(mode="w+", delete=False, encoding="utf-8")
    _config.transient.reportfile = report
    _config.transient.report_filename = report.name
    forward = FakeTranslator("en", "pt")
    monkeypatch.setattr(
        garak.probes.base.Probe, "_get_langprovider", lambda self: forward
    )
    monkeypatch.setattr(
        garak.probes.base.Probe,
        "_get_reverse_langprovider",
        lambda self: FakeTranslator("pt", "en"),
    )
    yield forward
    report.close()


class _LocalizingProbe(garak.probes.base.Probe):
    lang = "en"
    goal = "check localized prompts"
    primary_detector = "always.Pass"
    active = False

    def __init__(self, config_root=_config):
        super().__init__(config_root=config_root)
        plain = ["hidden payload"]
        localized = self._localize(plain)
        self.prompts = [
            self._localized_message(t.upper(), s.upper())
            for t, s in zip(localized, plain)
        ] + [
            Conversation(
                [
                    Turn("user", self._localized_message(localized[0], plain[0])),
                    Turn("assistant", Message("plain turn", lang=self.lang)),
                ]
            )
        ]


def test_localized_prompts_are_not_translated_again(fake_pt):
    p = _LocalizingProbe()
    gen = _plugins.load_plugin("generators.test.Repeat")
    attempts = p.probe(gen)

    assert fake_pt.calls[0] == ["hidden payload"]
    assert attempts[0].prompt.last_message().text == "HIDDEN PAYLOAD EM PORTUGUÊS, AÇÃO"
    turns = attempts[1].prompt.turns
    assert turns[0].content.text == "hidden payload em português, ação"
    assert turns[1].content.text == "plain turn em português, ação"
    for attempt in attempts:
        assert attempt.lang == "pt"
        for turn in attempt.prompt.turns:
            assert "source_text" not in turn.content.notes


def test_report_keeps_the_source_text(fake_pt):
    p = _LocalizingProbe()
    gen = _plugins.load_plugin("generators.test.Repeat")
    attempts = p.probe(gen)

    source = attempts[0].notes["pre_translation_prompt"]
    assert source.last_message().text == "HIDDEN PAYLOAD"
    assert source.last_message().lang == "en"
    turns = attempts[1].notes["pre_translation_prompt"].turns
    assert [t.content.text for t in turns] == ["hidden payload", "plain turn"]


def test_no_translation_leaves_prompts_alone(monkeypatch):
    _config.load_base_config()
    p = _LocalizingProbe()
    assert not p._localizing()
    assert p._localize(["x"]) == ["x"]
    msg = p._localized_message("X", "x")
    assert msg.text == "X" and msg.lang == "en" and not msg.notes


class _Decoder(garak.generators.base.Generator):
    """Toy target that obeys: decodes whatever it is sent and says it back"""

    generator_family_name = "test"
    name = "decoder"

    def __init__(self, decode, config_root=_config):
        self.decode = decode
        super().__init__(self.name, config_root=config_root)

    def _call_model(self, prompt, generations_this_call=1):
        return [Message(self.decode(prompt.last_message().text))]


def _b64_decode_all(text):
    found = []
    for token in re.findall(r"[A-Za-z0-9+/_=-]{8,}", text):
        for decode in (base64.b64decode, base64.urlsafe_b64decode):
            try:
                found.append(decode(token).decode("utf-8"))
                break
            except ValueError:
                pass
    return "\n".join(found)


def _rot13(text):
    return text.translate(garak.probes.encoding.ROT13TRANS)


@pytest.mark.parametrize(
    "classname,decode",
    [("InjectBase64", _b64_decode_all), ("InjectROT13", _rot13)],
)
def test_obedient_decoder_is_caught_in_target_language(fake_pt, classname, decode):
    _config.plugins.probes["encoding"] = {"payloads": ["default"]}
    p = _plugins.load_plugin(f"probes.encoding.{classname}")
    assert all(m.lang == "pt" for m in p.prompts)
    assert all(t.endswith("em português, ação") for t in p.triggers)

    attempts = p.probe(_Decoder(decode))
    detector = _plugins.load_plugin("detectors.encoding.DecodeMatch")
    scores = [s for a in attempts for s in detector.detect(a)]
    assert scores and all(s == 1.0 for s in scores)


def test_ascii_only_encoding_gets_folded_payload(fake_pt):
    _config.plugins.probes["encoding"] = {"payloads": ["default"]}
    p = _plugins.load_plugin("probes.encoding.InjectMorse")
    folded = garak.resources.encodings.ascii_fold(p.triggers[0])
    assert folded != p.triggers[0]
    assert garak.probes.encoding.morse(folded.encode()).decode() in (
        p.prompts[0].text + "".join(m.text for m in p.prompts[1:])
    )


def test_ascii_fold():
    fold = garak.resources.encodings.ascii_fold
    assert fold("Ação – “não”… pão") == 'Acao - "nao"... pao'
    assert fold("plain ASCII") == "plain ASCII"


def test_badchars_perturbs_the_translated_payload(fake_pt):
    p = _plugins.load_plugin("probes.badchars.BadCharacters")
    sources = list(garak.payloads.load("harmful_behaviors").payloads)
    assert fake_pt.calls == [sources], "only the clean payloads are translated"

    invisible = [
        c for c in p.prompts if c.notes["bad_character_category"] == "invisible"
    ]
    assert invisible
    for conv in invisible:
        msg = conv.last_message()
        assert msg.lang == "pt"
        visible = "".join(ch for ch in msg.text if ch.isprintable())
        assert visible.endswith("em português, ação")

    attempts = p.probe(_plugins.load_plugin("generators.test.Repeat"))
    assert len(fake_pt.calls) == 1, "perturbed prompts must not be translated again"
    assert attempts[0].notes["pre_translation_prompt"].last_message().text in sources
