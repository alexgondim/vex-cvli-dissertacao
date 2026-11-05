# Protocolo da rodada de 07/10 (bloco 26)

Gravado em 2026-10-07 17:46:37, antes do primeiro ajuste na base.

Base: `analises/medicoes_2026-09/base_2018_2024/painel_bairro_mes_features.csv`, sha256 `2a8288006c2323fa2f4995b82afa7e079f1944db804a0091c6ffd01224101443`. Portão do item 1:
confere em todos os números (log `rodada.log`).

## a. Grade do XGBoost

As 36 configurações da busca de 24/09 (`congelado_2026-09-24/hiperparametros_busca.py`, grade `xgb`), na ordem
daquele programa, sem acrescentar nem tirar nenhuma:

| ordem | learning_rate | n_estimators | max_depth | min_child_weight | |
|---|---|---|---|---|---|
| 1 | 0,05 | 100 | 3 | 1 | |
| 2 | 0,05 | 100 | 3 | 5 | |
| 3 | 0,05 | 100 | 3 | 20 | |
| 4 | 0,05 | 100 | 6 | 1 | |
| 5 | 0,05 | 100 | 6 | 5 | |
| 6 | 0,05 | 100 | 6 | 20 | |
| 7 | 0,05 | 300 | 3 | 1 | |
| 8 | 0,05 | 300 | 3 | 5 | |
| 9 | 0,05 | 300 | 3 | 20 | |
| 10 | 0,05 | 300 | 6 | 1 | |
| 11 | 0,05 | 300 | 6 | 5 | |
| 12 | 0,05 | 300 | 6 | 20 | |
| 13 | 0,1 | 100 | 3 | 1 | |
| 14 | 0,1 | 100 | 3 | 5 | |
| 15 | 0,1 | 100 | 3 | 20 | |
| 16 | 0,1 | 100 | 6 | 1 | |
| 17 | 0,1 | 100 | 6 | 5 | |
| 18 | 0,1 | 100 | 6 | 20 | |
| 19 | 0,1 | 300 | 3 | 1 | |
| 20 | 0,1 | 300 | 3 | 5 | |
| 21 | 0,1 | 300 | 3 | 20 | |
| 22 | 0,1 | 300 | 6 | 1 | |
| 23 | 0,1 | 300 | 6 | 5 | |
| 24 | 0,1 | 300 | 6 | 20 | |
| 25 | 0,3 | 100 | 3 | 1 | |
| 26 | 0,3 | 100 | 3 | 5 | |
| 27 | 0,3 | 100 | 3 | 20 | |
| 28 | 0,3 | 100 | 6 | 1 | padrão da biblioteca |
| 29 | 0,3 | 100 | 6 | 5 | |
| 30 | 0,3 | 100 | 6 | 20 | |
| 31 | 0,3 | 300 | 3 | 1 | |
| 32 | 0,3 | 300 | 3 | 5 | |
| 33 | 0,3 | 300 | 3 | 20 | |
| 34 | 0,3 | 300 | 6 | 1 | |
| 35 | 0,3 | 300 | 6 | 5 | |
| 36 | 0,3 | 300 | 6 | 20 | |

A configuração padrão da biblioteca está na grade, na posição 28: eta 0,3, max_depth 6, min_child_weight
1 e 100 árvores. Os três primeiros foram lidos da configuração de uma árvore de 1 rodada ajustada em dado
inventado de 2 linhas; o número de árvores, de `XGBRegressor().get_num_boosting_rounds()`. Nenhuma linha da base foi
usada nessa leitura.

## b. Critério de escolha

Menor deviance de Poisson média nas 1.992 previsões das 12 dobras de 2023, com piso de 0,01 nas estimativas (o piso
vale só na deviance). Em caso de empate até a 6ª casa decimal, fica a configuração que vem antes na grade.

Dobras: para cada mês-alvo de 2023, treino com todas as linhas de alvo anterior àquele mês, desde jan/2019, e
previsão das 166 linhas do mês.

## c. Estimador, semente, versão e threads

- `XGBRegressor(objective="count:poisson", enable_categorical=True, random_state=42)`, com os 4 hiperparâmetros da
  grade; o resto no padrão da biblioteca. Um modelo só para a cidade.
- Semente 42.
- XGBoost 3.0.2.
- Threads: `n_jobs` no padrão (`nthread` 0 na configuração, isto é, todos os núcleos); a máquina tem 20.
- `bairro`, `aisp` e `risp` entram como categóricas nativas (pandas `Categorical`, com o conjunto inteiro de valores
  em todas as dobras), sem colunas indicadoras. No padrão da biblioteca, `max_cat_to_onehot` = 4: a `risp`, com 3
  categorias, é dividida uma categoria contra as outras; `bairro` (166) e `aisp` (16), por conjunto de categorias.
- Proporções e idade média ausentes continuam ausentes, sem preenchimento.

## d. Atributos, por bloco

Os 30 atributos, na ordem do dicionário (`dicionario_painel_features.csv`) em todos os modelos:

- criminal (16): `cvli_lag1`, `cvli_lag2`, `cvli_lag3`, `cvli_lag4`, `cvli_lag5`, `cvli_lag6`, `cvli_lag7`, `cvli_lag8`, `cvli_lag9`, `cvli_lag10`, `cvli_lag11`, `cvli_lag12`, `cvli_ma3`, `cvli_ma6`, `cvli_ma12`, `cvli_trend`
- temporal (4): `month_sin_alvo`, `month_cos_alvo`, `year_norm_alvo`, `time_index`
- espacial (3): `bairro`, `aisp`, `risp`
- perfil (7): `n_vitimas_hist12`, `prop_madrugada_hist12`, `prop_tarde_hist12`, `prop_noite_hist12`, `prop_fds_hist12`, `prop_masc_hist12`, `idade_media_hist12`

## e. Ordem das etapas

1. Portão da construção.
2. Este protocolo.
3. Busca dos hiperparâmetros nas 12 dobras de 2023 (36 configurações).
4. Ablação M1 a M4 com a configuração escolhida; o M4 tem de repetir a busca.
5. Sensibilidade territorial (M4, MA, MB e MC), com contribuições e tau-b de Kendall.
6. Congelamento (`CONGELADO.tsv`).
7. Modelo do corte fixo, nas 7.968 linhas de alvo de 2019 a 2022.
8. Contribuições e interações do modelo do corte fixo.
9. Tabelas dos 5 usos do SHAP.
10. Painel ligado às saídas da rodada.
11. Capturas.
12. Zip da Parte 1 e parada. Os alvos de 2024 não entram em métrica nesta parte.
