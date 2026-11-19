"""**PromptInject**

Basic prompt injections, including plaintext malicious commands into requests
for innocuous information.

Probes in this module should specifically implement parts of the PromptInject
framework.

Uses the PromptInject framework, https://openreview.net/forum?id=qiaRo_7Zmug,
to probe a system. A subset of the attacks is implemented here. The default
is to use a selection of that subset, because even the subset corresponds to
thousands of requests by default.

Further info:

* https://openreview.net/forum?id=qiaRo_7Zmug

"""

import json
import random
import sys
from types import SimpleNamespace

from garak.attempt import Attempt
from garak.data import path as data_path
from garak.exception import GarakException
import garak.probes
from garak import _config

hijacking_attacks = (
    "ignore-say ignore-print ignore-print-upper nevermind screaming-stop".split()
)
rogue_strings = "hate-humans kill-humans long-prompt".split()
_generation_params = (
    "temperature top_p frequency_penalty presence_penalty max_tokens".split()
)

this = sys.modules[__name__]


# Sentence-final punctuation that MT appends to standalone translations.
# Kept to terminal marks only: broader punctuation (brackets, quotes, commas,
# hyphens) can be legitimate trigger content and must be preserved.
_TRAILING_TRIGGER_PUNCT = frozenset(".!?…‼⁇⁈⁉。！？؟۔։።፧।")


def _strip_trailing_punctuation(text: str) -> str:
    """Strip trailing sentence punctuation from a translated trigger.

    MT services sentence-punctuate standalone translations (e.g. a trailing
    period), but the translated prompt and the model output carry the phrase
    without it, so the detector's substring match misses. Strip trailing
    sentence-final punctuation so the trigger still matches.
    """
    return text.rstrip("".join(sorted(_TRAILING_TRIGGER_PUNCT)))


def _native_pieces(lang: str) -> dict | None:
    """prompt pieces reviewed in lang, from data/promptinject/prompt_data_<lang>.json"""
    for tag in dict.fromkeys([lang, lang.split("-")[0]]):
        try:
            with open(
                data_path / "promptinject" / f"prompt_data_{tag}.json",
                encoding="utf-8",
            ) as f:
                return json.load(f)
        except (FileNotFoundError, GarakException):
            continue
    return None


def _native_prompt_data(prompt_data, pieces: dict):
    """prompt_data with its texts replaced by the reviewed pieces

    Prompts are built from these pieces as from the English ones, so every
    combination is in the target language and nothing is translated on the fly.
    """
    bases = []
    for base in prompt_data.ALL_OPENAI_PLAYGROUND_PROMPTS:
        bases.append(base | pieces["base"][base["label"]])
    attacks = {
        name: attack | {"instruction": pieces["attacks"][name]}
        for name, attack in prompt_data.goal_hikacking_attacks.items()
    }
    return SimpleNamespace(
        ALL_OPENAI_PLAYGROUND_PROMPTS=tuple(bases),
        goal_hikacking_attacks=attacks,
        rogue_strings=pieces["rogue_strings"],
        escape_chars=prompt_data.escape_chars,
        delimiter_chars=prompt_data.delimiter_chars,
    )


def constructor(self, config_root=_config):
    super(self.__class__, self).__init__(config_root=config_root)
    from garak.resources.promptinject import (
        build_prompts,
        prompt_data,
    )
    from garak.detectors.ptbr import response_lang

    self.max_prompts = self.soft_probe_prompt_cap
    if self.__class__.__name__.endswith("Full"):
        self.max_prompts = None

    english_rogue_string = prompt_data.rogue_strings[self.rogue_string]
    pieces = (
        _native_pieces(self.langprovider.target_lang) if self._localizing() else None
    )
    if pieces is not None:
        # prompts built from pieces reviewed in the target language go as they are
        prompt_data = _native_prompt_data(prompt_data, pieces)
        self.lang = self.langprovider.target_lang

    # the injected string also counts in the original and in the answer language
    self._extra_triggers = [english_rogue_string]
    answer_lang = response_lang(self.lang)
    if answer_lang.split("-")[0] != self.lang.split("-")[0]:
        answer_pieces = _native_pieces(answer_lang)
        if answer_pieces is not None:
            self._extra_triggers.append(answer_pieces["rogue_strings"][self.rogue_string])

    self.pi_build_prompts = build_prompts
    self.load_data_all_attacks(prompt_data)

    # Pre-translate this probe's rogue string when prompts are localised: the
    # rogue string is translated inside the sent prompt, so the trigger stored
    # in attempt notes must be the translated form for the detector to score
    # against it. Mirrors the trigger translation in Probe.__init__ in
    # garak.probes.base, which only covers class-level triggers and not the
    # per-attempt notes set in _attempt_prestore_hook.
    self._translated_rogue_strings = {}
    if self.langprovider is not None and self.langprovider.target_lang != self.lang:
        rogue_string = prompt_data.rogue_strings[self.rogue_string]
        translated = self.langprovider.get_text([rogue_string])[0]
        # Standalone MT output carries sentence-final punctuation that the
        # translated prompt and model output lack; that alone breaks the
        # detector's substring match, so the trigger is stored stripped.
        # Never store an empty trigger: it is a substring of every output.
        stripped = _strip_trailing_punctuation(translated)
        self._translated_rogue_strings[rogue_string] = stripped or translated

    # self.prompts.append(f"{prefix} {false_assertion}")


def load_data_all_attacks(self, prompt_data):
    prompt_config = {
        "base": {
            "text": prompt_data.ALL_OPENAI_PLAYGROUND_PROMPTS,
        },
        "attack": {
            "text": (
                prompt_data.goal_hikacking_attacks["ignore-print"],
                prompt_data.goal_hikacking_attacks["ignore-say"],
                prompt_data.goal_hikacking_attacks["ignore-print-upper"],
                prompt_data.goal_hikacking_attacks["nevermind"],
                prompt_data.goal_hikacking_attacks["screaming-stop"],
            ),
            "rogue_string": prompt_data.rogue_strings[self.rogue_string],
            "scoring": "match_rogue_to_output",
            "settings": {
                "escape": prompt_data.escape_chars["n"],
                "delimiter": prompt_data.delimiter_chars["dash"],
                "escape_length": 10,
                "escape_times": 2,
            },
        },
    }
    self.pi_prompts = self.pi_build_prompts(prompt_config)
    if self.max_prompts:
        # sample pi_prompts, not just prompts, so seq indexes the same entry in both
        random.seed(self.seed)
        random.shuffle(self.pi_prompts)
        self.pi_prompts = self.pi_prompts[-self.max_prompts :]
    self.prompts = [pi_prompt["prompt"] for pi_prompt in self.pi_prompts]


def _attempt_prestore_hook(self, attempt: Attempt, seq: int) -> Attempt:
    attempt.notes["settings"] = self.pi_prompts[seq]["settings"]
    rogue_string = self.pi_prompts[seq]["settings"]["attack_rogue_string"]
    # Score against the translated rogue string when prompts are localised;
    # fall back to the English string if no translation was cached. The
    # original string is tracked as well: a detector working on
    # back-translated output needs the source form the prompt was based on.
    translated = self._translated_rogue_strings.get(rogue_string)
    if translated is not None and translated != rogue_string:
        triggers = [translated, rogue_string]
    else:
        triggers = [rogue_string]
    triggers += getattr(self, "_extra_triggers", [])
    attempt.notes["triggers"] = list(dict.fromkeys(triggers))
    return attempt


def _generator_precall_hook(self, generator, attempt):
    """apply PromptInject model inference parameters to probe"""
    map_attribs = self.generation_params
    for map_attrib in map_attribs:
        if map_attrib in dir(generator):
            setattr(
                generator,
                map_attrib,
                attempt.notes["settings"]["config_" + map_attrib],
            )


for rogue_string in rogue_strings:
    classname = "Hijack" + rogue_string.replace("-", " ").title().replace(" ", "")

    setattr(
        this,
        classname + "Full",
        type(
            classname + "Full",
            (garak.probes.Probe,),
            {
                "__init__": constructor,
                "__doc__": f"PromptInject probe to inject a string; payload is {rogue_string}\n\nAttempt to make the model repeat a given string",
                "name": "promptinject_" + rogue_string,
                "lang": "en",
                "description": "PromptInject is a framework that assembles prompts in a modular fashion to provide a quantitative analysis of the robustness of LLMs to adversarial prompt attacks. trophy Best Paper Awards @ NeurIPS ML Safety Workshop 2022",
                "intent": "T009ignore",  # PromptInject attacks try to override the model's current task and follow the injected one instead.
                # T008input is a secondary fit because success is often observed as verbatim repetition of the rogue string.
                "primary_detector": "promptinject.AttackRogueString",
                "parallelisable_attempts": False,
                "tags": [
                    "avid-effect:security:S0403",
                    "owasp:llm01",
                    "quality:Behavioral:ContentSafety:Violence",
                    "quality:Security:PromptStability",
                    "payload:unwanted:violence",
                    "demon:Language:Prompt_injection:Strong_arm_attack",  # includes forceful attack variants like STOP EVERYTHING!!! NOW!!!
                    "demon:Language:Stylizing:Capitalizing",  # attack set includes uppercase override commands such as IGNORE ANY PREVIOUS...
                    "demon:Language:Prompt_injection:Ignore_previous_instructions",  # injects "ignore previous" commands into benign prompt contexts
                    "euai:harmful:violence",
                    "euai:robustness:adversarial",
                ],
                "load_data_all_attacks": load_data_all_attacks,
                "uri": "https://openreview.net/forum?id=qiaRo_7Zmug",
                "rogue_string": rogue_string,
                "_attempt_prestore_hook": _attempt_prestore_hook,
                "_generator_precall_hook": _generator_precall_hook,
                "goal": "inject a prompt on " + rogue_string.replace("-", " "),
                "active": False,
                "tier": garak.probes.Tier.COMPETE_WITH_SOTA,
                "DEFAULT_PARAMS": garak.probes.Probe.DEFAULT_PARAMS
                | {"generation_params": _generation_params},
            },
        ),
    )

    setattr(
        this,
        classname,
        type(
            classname,
            (garak.probes.Probe,),
            {
                "__init__": constructor,
                "__doc__": f"PromptInject probe to inject a string; payload is {rogue_string}\n\nAttempt to make the model repeat a given string",
                "name": "promptinject_" + rogue_string,
                "lang": "en",
                "description": "PromptInject is a framework that assembles prompts in a modular fashion to provide a quantitative analysis of the robustness of LLMs to adversarial prompt attacks. trophy Best Paper Awards @ NeurIPS ML Safety Workshop 2022",
                "intent": "T009ignore",  # PromptInject attacks try to override the model's current task and follow the injected one instead.
                # T008input is a secondary fit because success is often observed as verbatim repetition of the rogue string.
                "primary_detector": "promptinject.AttackRogueString",
                "parallelisable_attempts": False,
                "tags": [
                    "avid-effect:security:S0403",
                    "owasp:llm01",
                    "quality:Behavioral:ContentSafety:Violence",
                    "quality:Security:PromptStability",
                    "payload:unwanted:violence",
                    "demon:Language:Prompt_injection:Strong_arm_attack",  # includes forceful attack variants like STOP EVERYTHING!!! NOW!!!
                    "demon:Language:Stylizing:Capitalizing",  # attack set includes uppercase override commands such as IGNORE ANY PREVIOUS...
                    "demon:Rhetoric:Persuasion_and_manipulation:Distraction",  # hides the injection inside otherwise benign Playground prompt contexts
                    "demon:Language:Prompt_injection:Ignore_previous_instructions",  # injects "ignore previous" commands into benign prompt contexts
                    "euai:harmful:violence",
                    "euai:robustness:adversarial",
                ],
                "load_data_all_attacks": load_data_all_attacks,
                "uri": "https://openreview.net/forum?id=qiaRo_7Zmug",
                "rogue_string": rogue_string,
                "_attempt_prestore_hook": _attempt_prestore_hook,
                "_generator_precall_hook": _generator_precall_hook,
                "goal": "inject a prompt on " + rogue_string.replace("-", " "),
                "active": True,
                "tier": garak.probes.Tier.COMPETE_WITH_SOTA,
                "DEFAULT_PARAMS": garak.probes.Probe.DEFAULT_PARAMS
                | {"generation_params": _generation_params},
            },
        ),
    )
