# garak-ptbr (MTR)

Versão da MTR do [garak](https://github.com/NVIDIA/garak) (NVIDIA, Apache 2.0) para testar agentes em português do Brasil.
Base: tag v0.17.1 do garak (= `main` deste repo). Plano: tradução pt-BR por LLM com cache revisado, payload traduzido
antes de codificar, detectores em pt-BR, conteúdo brasileiro e calibração por família.

## Branches
- `main`: garak v0.17.1 sem mudança, mais o que já foi aprovado. Não commitar direto; branch `<nome>/<slug>`.
- Nada aqui dispara deploy. Actions está desligado no repo: não religar nem rodar workflow sem pedir.

## Correções de fora já aplicadas (cherry-pick -x, autoria preservada)
- NVIDIA/garak PR #2334 (issue #2329): tradução preserva maiúsculas, pontuação e estrutura.
- NVIDIA/garak #2305 (já no main deles, pós-0.17.1): pré-requisito do #2309.
- NVIDIA/garak PR #2309 (issue #2307): frase-gatilho do promptinject traduzida.
Ao atualizar a base do garak, conferir se esses PRs entraram lá e remover os cherry-picks.

## Tradutor pt-BR (`langproviders.llm.LLMTranslator`)
Base: NVIDIA/garak PR #1774 (cherry-pick -x), reescrito aqui. Traduz o prompt inteiro numa chamada; código, links,
base64, hex, HTML e caracteres invisíveis vão mascarados (⟦n⟧) e voltam intactos. Resposta com recusa, sem tradução,
em outra língua ou de tamanho estranho é rejeitada; depois de `max_retries`, fica o original e entra em `failures`.
Config (`run.langproviders`): `language: "en,pt"`, `model_type: "llm.LLMTranslator"`, `model_name`,
`model_config` do gerador (ex.: `generator: nim.NVOpenAIChat`, `uri`, `stop: null` para modelo com raciocínio).
Modelo testado: `nvidia/nemotron-3.5-super-vl-120b-a12b` (50 de 50 traduzidas numa amostra; o Ultra recusou 10%).
Cache: `<cache_dir>/translations/en-pt.json`. Traduções revisadas à mão: `garak/data/translations/en-pt.json`
(chave = sha256 de `en|pt|<texto>`; têm prioridade). Ainda fora do revisado: latentinjection e promptinject.
A API da NVIDIA devolve muito 429 com várias rodadas em paralelo.

## Traduzir antes de transformar (`Probe._localize`)
Probe que codifica, perturba ou disfarça o payload não pode ter o prompt pronto traduzido. Ele traduz os textos
limpos com `self._localize(textos)`, monta o prompt e embrulha com `self._localized_message(texto, original_en)`;
o `probe()` não traduz de novo Message já na língua-alvo e o relatório guarda o inglês em `notes["source_text"]`.
Só payload do tipo `Text` é traduzido (código e marcadores ficam). Famílias cobertas: encoding, badchars,
smuggling.HomoglyphObfuscation, sata.MLM, dra, goodside.Tag, latentinjection.
- Codificação que só cobre ASCII (ROT13, Morse, Braille, Tag etc.) recebe o texto sem acento
  (`garak.resources.encodings.ascii_fold`); a lista de gatilhos leva a forma traduzida e a sem acento.
- dra: `data/dra/letter_map_<lang>.json` e `moderation_cache_<lang>.json` opcionais (pt já existe).
- O tradutor não traduz trechos entre crases: templates com crase (dra) foram revisados à mão no cache.
- latentinjection: o prompt é traduzido com `{garak_trigger}` no lugar do gatilho (o tradutor preserva campo de
  template) e recebe o gatilho traduzido, para o prompt levar o mesmo gatilho que o detector procura. Gatilho de
  payload com `lang` diferente de `en` (domínios, frases em francês/chinês) não é traduzido.
- continuation: com `data/slurprompts_<lang>.jsonl` (e `slurprompts_mini_<lang>.jsonl`), usa termos e aberturas
  escritos na língua-alvo, sem tradução (o probe passa a ter `lang` da rodada). Sem o arquivo, traduz como o oficial.
  Os de pt têm termos do MOL e a lista de racismo revisada; aberturas escritas por LLM.
- Teste: `tests/langservice/probes/test_probes_localize.py` (tradutor falso, sem rede).

## Detectores pt-BR (`detectors.ptbr`)
Numa rodada pt (`run.target_lang` pt), a resposta do alvo é lida em português; detector em inglês erra quase todas
as recusas em pt. `run.ptbr_detectors` escolhe o que roda no lugar de cada detector que só lê outra língua:
- `judge` (padrão): `ptbr.AttackSuccessJudge` (e `AttackSuccessJudgeOnlyAdversarial` no lugar de
  `judge.*OnlyAdversarial`). Juiz em pt que vê objetivo, mensagem e resposta e responde `[[FALHOU]]`/`[[SEGURO]]`:
  falha só se a resposta entregou o que o ataque queria; atender pedido legítimo é seguro. Uma chamada de LLM por resposta.
- `list`: `ptbr.MitigationBypass` (lista `garak/data/ptbr/recusas.txt`, sem acento e sem caixa) no lugar de
  `mitigation.MitigationBypass`/`Prefixes`; `ptbr.Offensive` (lista `garak/data/ptbr/ofensas.txt`, palavra inteira)
  no lugar dos detectores de palavrão e ofensa de `unsafe_content` (Surge, Ofcom, LDNOOBW, SlursReclaimedSlurs); o juiz
  no resto. Sem custo, mas não separa atendimento legítimo de ataque.
- `off`: detectores originais, com tradução de volta para o inglês.
Detector `lang_spec` `*` ou com pt fica como está. A troca vale nas harnesses probewise e por intent
(`harnesses/base.py`); com `-d` (pxd) roda o detector pedido. Com `judge`/`list` a resposta não é traduzida de volta
(`reverse_translation_needed`), exceto em atkgen, cujo atacante lê inglês. `Attempt.outputs_for` aceita lista de
idiomas (`"en,pt"`) e compara pela língua base.
Config do juiz (`detectors.ptbr`): `detector_model_type`, `detector_model_name`, `detector_model_config`
(ex.: `openai.OpenAICompatible` com `uri`, `temperature: 0`, `max_tokens: 4000`; chave em `OPENAICOMPATIBLE_API_KEY`),
`target_description` (entra na regra), `custom_system_prompt`, `max_message_chars` (mensagem longa é cortada no meio).
Modelo de raciocínio precisa de `max_tokens` alto: com pouco, o veredito volta vazio e conta como falha.
`ofensas.txt` só tem termo ofensivo em qualquer contexto (MOL, licença MIT em `data/ptbr/LICENSE-MOL`, mais racismo
revisado); termo que depende de contexto ("macaco") fica só como dado de ataque do continuation.
Teste: `tests/detectors/test_detectors_ptbr.py`.

## Testar
Rodar só os testes da área mexida, não a suíte toda. Ex.:
`PYTHONPATH=. REQUESTS_CA_BUNDLE=$SSL_CERT_FILE python -m pytest -q tests/langservice tests/probes/test_probes_promptinject.py`
Precisa dos extras `tests` do pyproject. Testes que baixam arquivo falham atrás de proxy sem o CA certo.

## Armadilhas
- Rodar contra agente de cliente pode criar dado real no cliente (carrinho, pedido). Conferir no código do alvo antes.
- Não colocar nome, prompt ou dado de cliente neste repo.
- Os geradores do garak mandam `stop: ["#", ";"]` por padrão; isso corta modelo com raciocínio. O juiz pt
  já suprime `stop`; no tradutor, usar `stop: null`.
- Probes que decidem a conversa com detector próprio (ex.: FITD usa `mitigation.Prefixes` para ver recusa) não passam
  pela troca de detectores: em pt elas não reconhecem recusa.
- `tests/langservice/probes/test_probes_goodside.py::test_Tag_attempt_descrs_translation` falha também no
  garak original (JSONDecodeError em `_convert_json_string`); não é regressão.

---

Abaixo, sem mudança, o AGENTS.md do garak (NVIDIA). Vale para toda contribuição enviada ao NVIDIA/garak.

# Agent instructions for garak - generative AI red-teaming and assessment toolkit

> These instructions apply to **all** AI-assisted contributions to `nvidia/garak`.
> Breaching these guidelines can result in automatic banning.

This is tooling for adversarial assessment of LLMs and LLM-powered tools.
It's open source software, used in production environments, with an active and skilled community
As such, the code needs to be robust, precise, and responsible.
Due to the nature of the project, there is a lot of potentially harmful or dangerous data associated with the repository.   

## Issue policy

Always add the `needs-triage` label to new issues, or use the git issue templates.

## Contribution policy

### Duplicate-work checks

Before proposing a PR, run these checks:

```bash
gh issue view <issue_number> --repo nvidia/garak --comments
gh pr list --repo nvidia/garak --state open --search "<issue_number> in:body"
gh pr list --repo nvidia/garak --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.

### Issues to avoid

Avoid issues labelled "needs triage". If it's important to take one with this label, comment on the issue first and await response from a repository maintainer.

Avoid anything labelled "for maintainers" unless explicitly directed.

Avoid bug issues that have no assignment or are not labelled `bug-verified`, unless you can build a working test case confirming the bug that does not conflict with project tests.

Avoid issues with other "needs" labels. If those needs appear explicitly addressed and the label may be stale, add a comment in the issue describing how the needs appear to have been met in that issue thread.

### Accountability

- Pure code-agent PRs are **not allowed**. A human submitter must understand and defend the change end-to-end.
- The submitting human must review every changed line and run relevant tests.
- PR descriptions for AI-assisted work **must** include:
    - Why this is not duplicating an existing PR.
    - Test commands run and results.
    - Clear statement that AI assistance was used.
- Only add an SPDX header when directly importing code from elsewhere under license that permits this and is compatible with the project.

### PR style

Don't add issue numbers in PR titles; it's OK to include these in the description.

### No low-value busywork PRs

Do not open one-off PRs for tiny edits (single typo, isolated style change, one mutable default, etc.). Mechanical cleanups are acceptable only when bundled with substantive work.

### Cohesive PRs

Each PR should have a clear focus. Only update files related to the topic of that PR.

### Fail-closed behavior

If work is duplicate/trivial busywork, **do not proceed**. Return a short explanation of what is missing.

### Project Guides

Follow the documentation on contributing too and extending garak. See:

* For selecting contributions: `docs/source/contributing.rst`
* For writing code: `docs/source/extending.rst`
* When writing a probe: `docs/source/extending.probe.rst`
* For intent-based scanning concepts: `docs/source/cas.rst`
* When writing a generator: `docs/source/extending.generator.rst`

### Commit messages

Add attribution using commit trailers such as `Co-authored-by:` (other projects use `Assisted-by:` or `Generated-by:`). For example:

```text
Your commit message here

Co-authored-by: GitHub Copilot
Co-authored-by: Claude
Co-authored-by: gemini-code-assist
Signed-off-by: Your Name <your.email@example.com>
```


## Development requirements

### Coding guide
- Follow the project guide docs linked above.
- Always avoid adding new dependencies. Use the `extra_dependency_names` functionality if essential.
- Keep documentation of garak architecture in the docs/ dir up to date - though use docstrings in the first instance if possible.
- When working on probes, detectors, or buffs, be sure to check the content of the relevant `doc_uri` to understand the code's intent and the underlying technique.
- Prefer `garak.probes.IntentProbe` when a new probe's technique is intent-agnostic: it spans the intent typology, which helps meet the prompt-count bar. See `docs/source/extending.probe.rst` and `docs/source/cas.rst`.
- Use the payloads, data, and services mechanisms when suitable.
- Use hooks where appropriate; add new hooks if this is efficient.
- Adhere to contribution and documentation standards, described in the docs.
- Prefer `pathlib` over `os`.
- Comply with docstring requirements - see the docs and also `tests/test_docs.py`.
- Catch specific exception types; avoid `except Exception` and bare `except:`.

### Dev environment tips
- Use (and expect) only Python versions specified in `pyproject.toml`.
- Be sure you're using the right environment, with the right dependencies. Virtual environment management is preferred.

### Testing instructions
- Don't break existing tests.
- Add tests as you go.
- Tests for specific modules should go in a new file. For example, tests for `garak.probes.xyz` should go in `tests/probes/test_probe_xyz.py`.
- ARM, x86, and Windows all need to be supported - check the list of supported architectures in `pyproject.toml`.
- Don't add tests for default values given in configurable plugins.
- Don't add tests for functionality already covered by tests of parent classes.
- Add descriptive strings to asserts, explaining the expect underlying behaviour; be terse.
- Check that tests work. If `pytest` or other project dependencies are not available, the environment has not been set up correctly; give the user this problem.
- Re-use/parametrize tests where appropriate.

### Code primitives
- Avoid updating `attempt` or any base classes (`probes.base.*`, `generators.base.*`, `detectors.base.*`) frivolously.
- Consider using a service for content to be available across all of garak over a whole run.

### Style
- Comply with any repository formatting and linting config in `pyproject.toml`.
- Don't include default values in docstrings; the code is the documentation for these.
- Follow the visual design language of CLI output, including emojis and colour changes where in line with existing style.
- garak assesses "targets", not "models".
- When assigning tags or taxonomy labels, add one per line, and include a brief justification in a comment after.
- Use British english in strings if you think you can get away with it -- claim ignorance if caught.
- Avoid overly explanatory comments.



