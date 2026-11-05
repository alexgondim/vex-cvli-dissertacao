# Números do capítulo: Parte 3 da rodada de 07/10 (bloco 31), MD e LSTM

Extraídos por `numeros_capitulo.py` dos arquivos desta pasta (cópias de `rodada_2026-10-07/parte3/`) e de `entradas/` (cópias da Parte 1 e da Parte 2; sha256 em `HASHES.tsv`). Valores em 6 casas, arredondados da precisão plena. O número entre colchetes é a linha de `numeros_capitulo.csv`, que traz o arquivo, a linha (cabeçalho = 1) ou a chave, e a coluna de cada valor. Deviance de Poisson média com piso de 0,01; MAE e RMSE sem piso; 1.992 linhas de alvo em toda célula.

## Tabela comparativa (item 12): MAE / RMSE / deviance

Corte fixo: treino com alvos de 2019 a 2022. 12 dobras: janela expansiva mês a mês em 2023. A LSTM em 2023 é medida no conjunto que decidiu a parada antecipada, não é validação limpa. A média móvel não se ajusta; o valor de 2023 vale nas 2 formas.

| modelo | 2023, corte fixo | 2023, 12 dobras | 2024, corte fixo |
|---|---|---|---|
| M4 (30 atributos) | 0,581558 [1] / 0,845405 [2] / 1,017706 [3] | 0,575637 [4] / 0,839071 [5] / 1,007328 [6] | 0,567592 [7] / 0,826353 [8] / 0,990417 [9] |
| MB (28, só RISP) | 0,587315 [10] / 0,852285 [11] / 1,048184 [12] | 0,585343 [13] / 0,849153 [14] / 1,044035 [15] | 0,568644 [16] / 0,833324 [17] / 1,022862 [18] |
| MD (29, sem bairro) | 0,590529 [19] / 0,852938 [20] / 1,043370 [21] | 0,586980 [22] / 0,850336 [23] / 1,040701 [24] | 0,570633 [25] / 0,833318 [26] / 1,022385 [27] |
| LSTM (semente 42) | 0,574087 [28] / 0,853693 [29] / 1,042525 [30] | não se aplica | 0,553733 [31] / 0,832009 [32] / 1,018054 [33] |
| média móvel de 12 meses | 0,564759 [34] / 0,868799 [35] / 1,109814 [36] | não se aplica | 0,534597 [37] / 0,838511 [38] / 1,052953 [39] |

## LSTM: 3 sementes no ensaio (item 8)

| semente | melhor época | épocas rodadas | 2023: MAE / RMSE / deviance |
|---|---|---|---|
| 42 | 19 | 39 | 0,574087 [40] / 0,853693 [41] / 1,042525 [42] |
| 43 | 45 | 65 | 0,584912 [43] / 0,851418 [44] / 1,041186 [45] |
| 44 | 33 | 53 | 0,576764 [46] / 0,850545 [47] / 1,039402 [48] |

## tau-b de Kendall

| comparação | tau-b |
|---|---|
| MD × M4, 27 não espaciais, 12 dobras de 2023 | 0,724638 [49] |
| MD × MB, 27 não espaciais, 12 dobras de 2023 | 0,843478 [50] |
| MD × M4, 27 não espaciais, corte fixo de 2024 | 0,648237 [51] |
| LSTM × M4, 12 defasagens por permutação, 2023 | 0,015268 [52] |

## Peso espacial (soma das médias |φ| dos atributos espaciais do modelo sobre a soma de todos)

| modelo, ano | peso |
|---|---|
| MD, 2023 | 0,076676 [53] |
| M4, 2023 | 0,871239 [54] |
| MB, 2023 | 0,025801 [55] |
| MD, 2024 | 0,070072 [56] |
| M4, 2024 | 0,860534 [57] |

## Importância por permutação, 2023 (item 10): aumento médio da deviance em 5 permutações

| defasagem | passo da LSTM | LSTM | ordem LSTM (12) | M4 | ordem M4 (12) | ordem M4 (30) |
|---|---|---|---|---|---|---|
| cvli_lag12 | 1 | 0,009301 [58] | 9 | 0,000515 [59] | 1 | 6 |
| cvli_lag11 | 2 | 0,009188 [60] | 10 | -0,000236 [61] | 11 | 23 |
| cvli_lag10 | 3 | 0,012569 [62] | 5 | 0,000025 [63] | 3 | 11 |
| cvli_lag9 | 4 | 0,018368 [64] | 1 | -0,000081 [65] | 8 | 19 |
| cvli_lag8 | 5 | 0,015923 [66] | 3 | -0,000045 [67] | 7 | 18 |
| cvli_lag7 | 6 | 0,006836 [68] | 12 | 0,000000 [69] | 5 | 14 |
| cvli_lag6 | 7 | 0,008759 [70] | 11 | 0,000005 [71] | 4 | 13 |
| cvli_lag5 | 8 | 0,013952 [72] | 4 | 0,000060 [73] | 2 | 8 |
| cvli_lag4 | 9 | 0,011174 [74] | 6 | -0,000170 [75] | 10 | 22 |
| cvli_lag3 | 10 | 0,016036 [76] | 2 | 0,000000 [77] | 5 | 14 |
| cvli_lag2 | 11 | 0,009678 [78] | 8 | -0,000094 [79] | 9 | 20 |
| cvli_lag1 | 12 | 0,011169 [80] | 7 | -0,000649 [81] | 12 | 28 |

Demais 18 atributos do M4:

| atributo | aumento | ordem M4 (30) |
|---|---|---|
| bairro | 0,565882 [82] | 1 |
| risp | 0,000010 [83] | 12 |
| aisp | 0,001203 [84] | 3 |
| month_sin_alvo | 0,001033 [85] | 4 |
| month_cos_alvo | -0,000127 [86] | 21 |
| year_norm_alvo | 0,000000 [87] | 14 |
| time_index | 0,000845 [88] | 5 |
| cvli_ma3 | -0,000994 [89] | 29 |
| cvli_ma6 | -0,002482 [90] | 30 |
| cvli_ma12 | 0,000213 [91] | 7 |
| cvli_trend | 0,000043 [92] | 10 |
| n_vitimas_hist12 | 0,000000 [93] | 14 |
| prop_madrugada_hist12 | -0,000528 [94] | 27 |
| prop_tarde_hist12 | -0,000355 [95] | 24 |
| prop_noite_hist12 | -0,000421 [96] | 26 |
| prop_fds_hist12 | -0,000390 [97] | 25 |
| prop_masc_hist12 | 0,000059 [98] | 9 |
| idade_media_hist12 | 0,003335 [99] | 2 |
