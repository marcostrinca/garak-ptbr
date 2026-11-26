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
  `stop_patterns`, `stop_file`) e o ritmo (`request_interval`). Se o chat mostra imagem ou botão a partir de um campo próprio da resposta,
  inclua o campo da URL em `response_json_field`: sem ele, o `web_injection` não vê o vazamento.
- `agent`: como o juiz pt chama o agente (`description`), a língua em que ele responde (`language`), o prompt de
  sistema real para o `sysprompt_extraction` (`system_prompt_file`; sem ele, essa família fica de fora) e a
  finalidade e as ferramentas para o `agent_breaker` (`purpose`, `tools`). `blocked_topics` (por idioma) são os
  assuntos de que o agente não deve falar: o `topic` desce no WordNet a partir deles (`topic.WordnetBlockedWords`).
  Sem a lista, o `topic` roda só os temas polêmicos de sempre.
  `image_format` é como a resposta escreve uma imagem que o chat mostra (`{uri}` é a URL; padrão: Markdown); o
  `web_injection.ImageExfil` pede a imagem nesse formato, com URL num domínio `.invalid`, que não existe.
- `run`: famílias e idiomas (`families`; sem isso, todas), objetivos do goat por idioma (`goat_goals_file`), alvos
  do fitd por idioma (`fitd_targets_file`, todos os da lista) e a pasta dos relatórios (`report_dir`).
- `model` (opcional): modelo do atacante, dos juízes e do tradutor. Padrão: DeepSeek-V4.1-Flash na together.ai.
- `garak` (opcional): qualquer config do garak, aplicada por último. No ensaio, ela não troca o alvo falso.

## Travas

- Se o `stop_file` existir, nenhuma rodada começa e nenhuma mensagem sai. Ele é criado quando um `stop_patterns`
  aparece numa resposta. Olhe o que aconteceu antes de apagar.
- Ponha em `stop_patterns` também a resposta que o agente dá quando não roda o modelo (limite por hora, sem saldo,
  fora do ar). Sem isso, a rodada segue e o juiz conta o aviso como recusa, "passou".
- Agente com limite de mensagens por hora: `request_interval` (segundos entre mensagens) abaixo do limite.
  Ex.: limite de 1.000 por hora, `request_interval: 4` (900 por hora).
- O `preflight_command` roda antes de cada mensagem com `preflight_every_request`; se não imprimir o esperado, nada
  é enviado.
- Ataque de vários turnos (goat, fitd, agent_breaker) age de verdade nas ferramentas do agente: num agente de loja,
  pôr na sacola grava dado. Rode contra um ambiente de teste.

## Rodada sem medida

Rodada que termina sem nenhuma nota sai como `NOT MEASURED` (e `--live` sai com código 3). Ela não diz nada sobre o
agente, nem que ele resistiu. Acontece, por exemplo, com o fitd contra um agente que recusa até o primeiro passo
inofensivo, ou quando o atacante não gera ataque. Leia o log do garak antes de repetir.
