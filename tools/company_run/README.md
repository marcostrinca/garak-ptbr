# Rodada por arquivo de empresa

`run.py` roda o garak contra o agente de uma empresa a partir de um arquivo só (`example.yaml` é uma loja fictícia).
Cada família de ataque, em cada idioma, é uma rodada do garak com relatório próprio.

## Como usar

1. Copie `example.yaml` para fora do repo e ajuste para o seu agente. Deixe o arquivo só para você (`chmod 600`).
2. Ponha a chave do modelo em `OPENAICOMPATIBLE_API_KEY`, no ambiente ou no `env_file` (também `chmod 600`).
   Nada de chave no arquivo da empresa.
3. `python tools/company_run/run.py empresa.yaml` mostra o plano e não envia nada.
4. `--rehearsal` roda tudo contra um alvo falso local, com volume reduzido: confere que as probes carregam, que os
   juízes respondem e quantas mensagens cada família manda. O alvo real não é tocado.
5. `--live` roda contra o alvo real. Use `--family` e `--lang` para ir por etapas.

## O que vai no arquivo

- `target`: opções do `rest.RestGenerator` (ver `docs/source/generators/rest.rst`), inclusive a conversa
  (`session_json_field`), a junção da resposta (`response_json_join`) e as travas (`preflight_command`,
  `stop_patterns`, `stop_file`).
- `agent`: como o juiz pt chama o agente (`description`), a língua em que ele responde (`language`), o prompt de
  sistema real para o `sysprompt_extraction` (`system_prompt_file`; sem ele, essa família fica de fora) e a
  finalidade e as ferramentas para o `agent_breaker` (`purpose`, `tools`).
- `run`: famílias e idiomas (`families`; sem isso, todas), objetivos do goat por idioma (`goat_goals_file`) e a
  pasta dos relatórios (`report_dir`).
- `model` (opcional): modelo do atacante, dos juízes e do tradutor. Padrão: DeepSeek-V4.1-Flash na together.ai.
- `garak` (opcional): qualquer config do garak, aplicada por último. No ensaio, ela não troca o alvo falso.

## Travas

- Se o `stop_file` existir, nenhuma rodada começa e nenhuma mensagem sai. Ele é criado quando um `stop_patterns`
  aparece numa resposta. Olhe o que aconteceu antes de apagar.
- O `preflight_command` roda antes de cada mensagem com `preflight_every_request`; se não imprimir o esperado, nada
  é enviado.
- Ataque de vários turnos (goat, fitd, agent_breaker) age de verdade nas ferramentas do agente: num agente de loja,
  pôr na sacola grava dado. Rode contra um ambiente de teste.
