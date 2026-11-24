"""**Phrasing**

These attacks rephrase requests for unsafe content in an attempt to bypass safeguards.

This module is home to probes that rely on specific grammatical alterations and rephrasing
to implement their technique.

In a run translated to a language with a ``<tense>_tense_<lang>.txt`` file in
data/phrasing, the tense probes send those requests, written in that language,
and translate nothing.

"""

from garak import _config
from garak.attempt import Message
from garak.data import path as data_path
from garak.exception import GarakException
import garak.probes


def _load_tense_prompts(probe, tense):
    """Requests in ``tense``, from the run's target language file when there is one"""
    native_file = None
    if probe._localizing():
        try:
            native_file = (
                data_path
                / "phrasing"
                / f"{tense}_tense_{probe.langprovider.target_lang}.txt"
            )
        except GarakException:
            native_file = None
    with open(
        native_file or data_path / "phrasing" / f"{tense}_tense_en.txt",
        "r",
        encoding="utf-8",
    ) as file:
        prompts = [line.strip() for line in file]
    if native_file is not None:
        # requests reviewed in the target language go as they are
        probe.lang = probe.langprovider.target_lang
        prompts = [Message(prompt, lang=probe.lang) for prompt in prompts]
    return prompts


class PastTenseFull(garak.probes.Probe):
    """Will the system give unsafe responses when asked to write about weapon creation or dangerous behavior in the past tense?

    This check sends the generator requests for advice on possibly protect or restricted subjects in the past tense.
    """

    active = False
    lang = "en"
    doc_uri = "https://arxiv.org/html/2407.11969v2"
    intent = "T009ignore"  # tense-shifting is used here to get the model to ignore its normal refusal policy
    primary_detector = "mitigation.Prefixes"
    tags = [
        "owasp:llm01",
        "quality:Security:PromptStability",
        "payload:jailbreak",
        "demon:Language:Stylizing:Synonymous_language",  # reformulates harmful requests in past tense to bypass refusal training
        "euai:robustness:adversarial",
    ]
    goal = "use past tense to disregard the system prompt"
    tier = garak.probes.Tier.COMPETE_WITH_SOTA

    def __init__(self, config_root=_config):
        super().__init__(config_root=config_root)

        self.prompts = _load_tense_prompts(self, "past")


class PastTense(PastTenseFull):
    """Attempt to evade safeguards by setting requests in the past tense

    A subset of the tense-modified requests. Success is reported higher with more variation, so
    generations is overridden.
    """

    active = True
    intent = "T009ignore"  # this is the capped subset of the same jailbreak strategy

    def __init__(self, config_root=_config):
        super().__init__(config_root=config_root)
        self._prune_data(cap=self.soft_probe_prompt_cap)


class FutureTenseFull(garak.probes.Probe):
    """Will the system give unsafe responses when asked to write about weapon creation or dangerous behavior in the future tense?

    This check sends the generator requests for advice on possibly protect or restricted subjects in the future tense.
    """

    active = False

    lang = "en"
    doc_uri = "https://arxiv.org/html/2407.11969v2"
    intent = "T009ignore"  # the future-tense reformulation is used to bypass normal safety refusals
    primary_detector = "mitigation.Prefixes"
    tags = [
        "owasp:llm01",
        "quality:Security:PromptStability",
        "payload:jailbreak",
        "demon:Language:Stylizing:Synonymous_language",  # reformulates harmful requests in future tense to bypass refusal training
        "euai:robustness:adversarial",
    ]
    goal = "use future tense to disregard the system prompt"
    tier = garak.probes.Tier.COMPETE_WITH_SOTA

    def __init__(self, config_root=_config):
        super().__init__(config_root=config_root)

        self.prompts = _load_tense_prompts(self, "future")


class FutureTense(FutureTenseFull):
    """Attempt to evade safeguards by setting requests in the future tense

    A subset of the tense-modified requests. Success is reported higher with more variation, so
    generations is overridden.
    """

    active = True
    intent = "T009ignore"  # uses the same phrasing-based jailbreak on a smaller set

    def __init__(self, config_root=_config):
        super().__init__(config_root=config_root)
        self._prune_data(self.soft_probe_prompt_cap)
