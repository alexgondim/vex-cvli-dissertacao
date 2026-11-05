# Protocolo da Parte 4 da rodada de 07/10 (bloco 37)

Gravado em 21/10/2026, antes de qualquer ajuste desta parte. A hora e o sha256 deste arquivo ficam no log (`parte4/rodada_parte4.log`), na primeira linha da validação, e no `CONGELADO_parte4.tsv`.

A ablação por blocos roda de novo com a composição pedida pelo orientador em 21/10. Nada da Parte 1, da Parte 2 nem da Parte 3 muda: as saídas desta parte ficam em `rodada_2026-10-07/parte4/`; o programa fica em `rodada_2026-10-07/`, junto dos programas da Parte 1, como manda o item 3 do bloco.

## 1. Composição dos blocos (item 1)

Vale para este bloco e para o capítulo. Escrita em `ablacao_blocos2.py` (`BLOCOS`), com o comentário de que substitui a da Parte 1 por pedido do orientador em 21/10.

- criminal (16): `cvli_lag1` a `cvli_lag12`, `cvli_ma3`, `cvli_ma6`, `cvli_ma12`, `cvli_trend`;
- temporal (8): `month_sin_alvo`, `month_cos_alvo`, `year_norm_alvo`, `time_index`, `prop_madrugada_hist12`, `prop_tarde_hist12`, `prop_noite_hist12`, `prop_fds_hist12`;
- espacial (3): `bairro`, `aisp`, `risp`;
- perfil (3): `prop_masc_hist12`, `idade_media_hist12`, `n_vitimas_hist12`.

Em relação à Parte 1 (`comum.BLOCOS`), as 4 proporções de turno e de fim de semana saem do perfil e entram no temporal; criminal e espacial são iguais. Antes de treinar, `conferir_blocos` confere por código: nenhum atributo repetido, 16, 8, 3 e 3 atributos, que juntos são os 30 do dicionário; criminal e espacial iguais aos da Parte 1; temporal igual ao da Parte 1 mais as 4 proporções; perfil igual ao da Parte 1 menos elas. Dentro de cada modelo, os atributos ficam na ordem do dicionário (`comum.em_ordem`), como na Parte 1.

## 2. Modelos (item 2)

| modelo | blocos | atributos |
|---|---|---|
| M1 | criminal | 16 |
| M2 | criminal + temporal | 24 |
| M3 | M2 + espacial | 27 |
| M4 | os 30 | 30 |

M1 e M4 têm os mesmos atributos dos M1 e M4 da Parte 1 e servem de conferência de reprodução. M2 e M3 são novos (na Parte 1, M2 tinha 20 atributos e M3, 23).

Os 4 usam a configuração 3 da grade, fixada na Parte 1 (`escolha.json`, lida pelo `comum2.CONFIG`): `learning_rate` 0,05, `n_estimators` 100, `max_depth` 3, `min_child_weight` 20. Estimador `XGBRegressor(objective="count:poisson", enable_categorical=True, random_state=42)` (`comum.estimador`), XGBoost 3.0.2; `bairro`, `aisp` e `risp` como categóricas nativas, com o conjunto inteiro de valores; proporções e idade média ausentes continuam ausentes. Não há busca de hiperparâmetro nem escolha nenhuma neste bloco.

Métricas (`comum.metricas`): MAE, RMSE e deviance de Poisson média, com piso de 0,01 na previsão só na deviance.

## 3. Programa (item 3)

`rodada_2026-10-07/ablacao_blocos2.py`, com 3 passos, nesta ordem: `validacao`, `congelar` e `teste`.

- **Reaproveitamento por import.** O programa importa o `comum2.py` (parte2/) e, por ele, o `comum.py` da Parte 1, de onde vêm a leitura da base conferida pelo sha256 (`comum.ler_base`, pelo `comum2.ler_base`), os 30 atributos (`comum.atributos`, `comum.em_ordem`), a configuração 3 (`comum2.CONFIG`), o estimador com a semente 42, as métricas, as dobras de 2023 e a previsão por dobra (`comum.prever_dobras`), os anos de treino do corte fixo (`comum2.ANOS_DE_TREINO`), o sha256 e a gravação sem sobrescrever (`comum2.gravar_csv` e `comum2.gravar_json`). Nenhum código é copiado. O SHAP não é usado neste bloco.
- **Nada importado do `ablacao_sensibilidade.py`.** Ele não tem funções: é um programa que, ao ser importado, roda de novo a ablação e a sensibilidade da Parte 1, regrava `ablacao.csv`, `previsoes_ablacao_sensibilidade.csv`, `contribuicoes_sensibilidade.csv` e os `sensibilidade_*.csv` e escreve no `rodada.log`. O que o item 3 pede dele (leitura da base, dobras, treino, métricas) está no `comum.py`, usado pelo `comum2.py`.
- **Mapa de blocos por parâmetro.** As funções da Parte 1 usadas aqui recebem a lista de atributos de cada modelo, e não o mapa de blocos; nenhuma depende da composição dos blocos. A única que lê o mapa antigo, `comum.atributos`, confere só a união dos blocos, que é a mesma nos 2 mapas (os 30 atributos). Por isso nenhum parâmetro foi acrescentado e nenhum arquivo da Parte 1 foi editado: `busca.py`, `ablacao_sensibilidade.py` e `comum.py` ficam com o sha256 do `CONGELADO.tsv` da Parte 1, e o mapa antigo continua neles como está. O mapa novo existe só no `ablacao_blocos2.py`.
- **Log próprio.** O `comum2.registrar` escreve no log da Parte 2 e não é usado; o log desta parte é `parte4/rodada_parte4.log`.
- **Sem bytecode.** O programa liga `sys.dont_write_bytecode` antes dos imports, para não gravar `__pycache__` nas pastas das Partes 1 e 2.
- **Sem sobrescrever.** Toda saída é gravada em modo de criação; um arquivo que já existe faz o programa parar.

## 4. Validação de 2023 (item 4), passo `validacao`

- Só roda com este protocolo gravado e sem `CONGELADO_parte4.tsv`.
- A base é lida pelo `comum2.ler_base("ensaio")`: o `comum.ler_base` confere o sha256 (`2a8288006c2323fa2f4995b82afa7e079f1944db804a0091c6ffd01224101443`) e as linhas de alvo em 2024 saem logo depois da leitura. Nenhuma linha de alvo em 2024 entra na validação.
- As 12 dobras da janela expansiva de 2023, as mesmas da Parte 1 (`comum.dobras_2023`): para cada mês-alvo de 2023, treino com todas as linhas de alvo anterior ao mês, desde jan/2019, e previsão das 166 linhas do mês.
- Para cada modelo: MAE, RMSE e deviance em cada dobra (166 linhas), a média das 12 dobras e as 3 métricas nas 1.992 linhas juntas.
- **Conferência de reprodução, critério fixado aqui:** M1 e M4 têm de repetir, escritos com 6 casas decimais (formato `%.6f`), o MAE, o RMSE e a deviance nas 1.992 linhas gravados pela Parte 1 em `rodada_2026-10-07/ablacao.csv` (sha256 `8bddcf420f9ed76cee9ef46e6924086b8eada462758fb3ceeacafddd0c717a65`):

  | modelo | MAE | RMSE | deviance |
  |---|---|---|---|
  | M1 | 0,587754 | 0,852464 | 1,048960 |
  | M4 | 0,575637 | 0,839071 | 1,007328 |

  Se não repetir, o programa para, o log registra a parada, e ela é reportada.
- **Ensaio do caminho do teste.** Antes do congelamento, a mesma função do teste (`corte_fixo`) roda com 2023 no lugar de 2024, só para M1 e M4: treino nas 7.968 linhas de alvo de 2019 a 2022 e avaliação nas 1.992 de 2023. Tem de repetir, com 6 casas, o ensaio da Parte 2 em `parte2/ensaio/ablacao_sensibilidade_2023.csv` (sha256 `df90457c54079650c75bf0a9ad8013e7c71493bd224e1080c9e638e3260aa936`): M1 0,588104, 0,853168 e 1,050205; M4 0,581558, 0,845405 e 1,017706 (MAE, RMSE e deviance). Serve só para conferir o programa antes de ver 2024; os números ficam no log. Se não repetir, o programa para; como nenhuma métrica de 2024 terá sido vista, o erro de programa é corrigido antes do congelamento, e a correção fica no log.
- Também antes do congelamento, a montagem da tabela de 2024 e do resumo (`tabela_do_teste` e `resumo_de`) é exercitada com números de 2023, fora do log e sem gravar nada, para que nenhuma parte do passo `teste` rode pela primeira vez com 2024.
- Grava `ablacao_blocos2_2023.csv`, `previsoes_ablacao_blocos2_2023.csv` e `conferencia_2023.json`.

## 5. Congelamento (item 5), passo `congelar`

Só roda com a conferência de 2023 fechada (`conferencia_2023.json` com `ok` verdadeiro), com `ablacao_blocos2_2023.csv` gravado e sem `ablacao_blocos2_2024.csv`. Grava `CONGELADO_parte4.tsv` com o sha256 de `parte4/protocolo_parte4.md`, `ablacao_blocos2.py` e `parte2/comum2.py`, com data e hora. O log registra que, até ali, nenhuma linha de alvo em 2024 foi usada e nenhuma métrica de 2024 foi calculada.

## 6. Teste (item 6), passo `teste`

- Só roda com o `CONGELADO_parte4.tsv` gravado e os 3 arquivos iguais aos congelados; confere também o `comum.py` contra o sha256 do `CONGELADO.tsv` da Parte 1 (`f121fa82c4957c4a7b27f827450fdb0949e8044a3c5fa892d2fe03387a6aa2d3`).
- Roda 1 vez: para se `ablacao_blocos2_2024.csv`, `previsoes_ablacao_blocos2_2024.csv` ou `resumo_parte4.json` já existir.
- Corte fixo: cada um dos 4 modelos treinado 1 vez nas 7.968 linhas de alvo de 2019 a 2022 e avaliado nas 1.992 linhas de 2024, com as 3 métricas.
- As saídas de 2024 são gravadas antes da conferência, para que nada precise rodar de novo.
- **Conferência de reprodução, critério fixado aqui:** M1 e M4 têm de repetir, com 6 casas, os valores da Parte 2 em `parte2/ablacao_sensibilidade_2024.csv` (sha256 `a65263c5328fda672efed683457b19cf43b84c55cf9778d8be1c0127ce19cd75`):

  | modelo | MAE | RMSE | deviance |
  |---|---|---|---|
  | M1 | 0,568209 | 0,834472 | 1,024669 |
  | M4 | 0,567592 | 0,826353 | 0,990417 |

  Se não repetir, o programa para sem gravar o resumo, e a parada é reportada, sem rodar de novo.
- Erro de programa descoberto depois de vista alguma métrica de 2024: parar e reportar, sem corrigir e sem rodar de novo.

## 7. Saídas (item 7)

- `ablacao_blocos2_2023.csv`: modelo, atributos, dobra, linhas, MAE, RMSE e deviance; para cada modelo, as 12 dobras (`2023-01` a `2023-12`), a linha `media_das_12_dobras` e a linha `todas_as_1992_linhas`.
- `ablacao_blocos2_2024.csv`: modelo, atributos, linhas, MAE, RMSE e deviance.
- `resumo_parte4.json`: a composição, os 4 modelos lado a lado em 2023 (média das 12 dobras e 1.992 linhas) e em 2024, e o ganho de cada bloco, que é a métrica do modelo com o bloco menos a do modelo anterior (negativo é melhora): temporal = M2 − M1, espacial = M3 − M2, perfil = M4 − M3.
- Para conferência: `previsoes_ablacao_blocos2_2023.csv`, `previsoes_ablacao_blocos2_2024.csv` e `conferencia_2023.json`.

## 8. Ordem de execução

1. Este protocolo.
2. `ablacao_blocos2.py validacao` (item 4), com a conferência de M1 e M4 e o ensaio do caminho do teste.
3. Exercício de `tabela_do_teste` e `resumo_de` com números de 2023, sem gravar nada.
4. `ablacao_blocos2.py congelar` (item 5).
5. `ablacao_blocos2.py teste` (item 6), 1 vez.
6. Depois do teste, só leitura das saídas: quadro comparativo com a Parte 1 (item 9), busca do mapa antigo fora da Parte 1 (item 8), pasta `congelado_2026-10-21/` com `HASHES.tsv` e a tabela "números do capítulo", relatório e zip.
