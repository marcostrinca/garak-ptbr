# garak-ptbr

Versão da MTR do [garak](https://github.com/NVIDIA/garak) (NVIDIA, Apache 2.0) para testar agentes em português do Brasil.
Base: tag v0.17.1 do garak (= `main` deste repo). Plano: tradução pt-BR por LLM com cache revisado, payload traduzido
antes de codificar, detectores em pt-BR, conteúdo brasileiro e calibração por família.

## Branches
- `main`: garak v0.17.1 sem mudança, mais o que já foi aprovado. Não commitar direto; branch `<nome>/<slug>`.
- Nada aqui dispara deploy. Actions herdados do garak estão ligados: não rodar workflow sem pedir.

## Correções de fora já aplicadas (cherry-pick -x, autoria preservada)
- NVIDIA/garak PR #2334 (issue #2329): tradução preserva maiúsculas, pontuação e estrutura.
- NVIDIA/garak #2305 (já no main deles, pós-0.17.1): pré-requisito do #2309.
- NVIDIA/garak PR #2309 (issue #2307): frase-gatilho do promptinject traduzida.
Ao atualizar a base do garak, conferir se esses PRs entraram lá e remover os cherry-picks.

## Testar
Rodar só os testes da área mexida, não a suíte toda. Ex.:
`PYTHONPATH=. REQUESTS_CA_BUNDLE=$SSL_CERT_FILE python -m pytest -q tests/langservice tests/probes/test_probes_promptinject.py`
Precisa dos extras `tests` do pyproject. Testes que baixam arquivo falham atrás de proxy sem o CA certo.

## Armadilhas
- Rodar contra agente de cliente pode criar dado real no cliente (carrinho, pedido). Conferir no código do alvo antes.
- Não colocar nome, prompt ou dado de cliente neste repo.
