# Números do capítulo: teste de 2024, Parte 2 da rodada de 07/10

Extraídos por `numeros_capitulo.py` de `metricas.csv` e `ablacao_sensibilidade_2024.csv` desta pasta, que são cópias das saídas de `rodada_2026-10-07/parte2/` (sha256 em `HASHES.tsv`). Valores em 6 casas, arredondados da precisão plena dos arquivos. O número entre colchetes é a linha de `numeros_capitulo.csv`, que traz o arquivo, a linha do arquivo (cabeçalho = 1) e a coluna de cada valor.

- Deviance de Poisson média, com piso de 0,01 na previsão; MAE e RMSE sem piso.
- Toda linha dos 2 quadros tem 1.992 linhas de alvo.
- No corte fixo, o modelo é o de 07/10, treinado com alvos de 2019 a 2022, sem reajuste; a binomial negativa é ajustada nas mesmas 7.968 linhas. Na janela expansiva, um reajuste por mês, de jan/2022 a dez/2024.

## Métricas por forma, ano e método (`metricas.csv`)

| forma | ano | método | MAE | RMSE | deviance de Poisson | linha do arquivo |
|---|---|---|---|---|---|---|
| corte fixo | 2024 | modelo | 0,567592 [1] | 0,826353 [2] | 0,990417 [3] | 2 |
| corte fixo | 2024 | binomial negativa | 0,576139 [4] | 0,843353 [5] | 1,058359 [6] | 3 |
| corte fixo | 2024 | persistência | 0,600402 [7] | 1,122068 [8] | 2,561716 [9] | 4 |
| corte fixo | 2024 | média móvel de 12 meses | 0,534597 [10] | 0,838511 [11] | 1,052953 [12] | 5 |
| corte fixo | 2024 | média sazonal | 0,570197 [13] | 0,894079 [14] | 1,154650 [15] | 6 |
| janela expansiva | 2022 | modelo | 0,599385 [16] | 0,887553 [17] | 1,015303 [18] | 7 |
| janela expansiva | 2022 | binomial negativa | 0,729089 [19] | 1,499798 [20] | 1,219961 [21] | 8 |
| janela expansiva | 2022 | persistência | 0,680723 [22] | 1,192345 [23] | 2,756077 [24] | 9 |
| janela expansiva | 2022 | média móvel de 12 meses | 0,601531 [25] | 0,920225 [26] | 1,104318 [27] | 10 |
| janela expansiva | 2022 | média sazonal | 0,612952 [28] | 0,975059 [29] | 1,439375 [30] | 11 |
| janela expansiva | 2023 | modelo | 0,575637 [31] | 0,839071 [32] | 1,007328 [33] | 12 |
| janela expansiva | 2023 | binomial negativa | 0,603755 [34] | 0,886032 [35] | 1,088012 [36] | 13 |
| janela expansiva | 2023 | persistência | 0,653614 [37] | 1,167670 [38] | 2,799974 [39] | 14 |
| janela expansiva | 2023 | média móvel de 12 meses | 0,564759 [40] | 0,868799 [41] | 1,109814 [42] | 15 |
| janela expansiva | 2023 | média sazonal | 0,604920 [43] | 0,938864 [44] | 1,405426 [45] | 16 |
| janela expansiva | 2024 | modelo | 0,550442 [46] | 0,814355 [47] | 0,969704 [48] | 17 |
| janela expansiva | 2024 | binomial negativa | 0,570949 [49] | 0,843970 [50] | 1,054799 [51] | 18 |
| janela expansiva | 2024 | persistência | 0,600402 [52] | 1,122068 [53] | 2,561716 [54] | 19 |
| janela expansiva | 2024 | média móvel de 12 meses | 0,534597 [55] | 0,838511 [56] | 1,052953 [57] | 20 |
| janela expansiva | 2024 | média sazonal | 0,570197 [58] | 0,894079 [59] | 1,154650 [60] | 21 |

## Ablação e sensibilidade territorial no corte fixo, 2024 (`ablacao_sensibilidade_2024.csv`)

Cada variante treinada nas 7.968 linhas de alvo de 2019 a 2022, com a configuração 3 da grade.

| variante | atributos | MAE | RMSE | deviance de Poisson | linha do arquivo |
|---|---|---|---|---|---|
| M1 | 16 | 0,568209 [61] | 0,834472 [62] | 1,024669 [63] | 2 |
| M2 | 20 | 0,567539 [64] | 0,833153 [65] | 1,023915 [66] | 3 |
| M3 | 23 | 0,566924 [67] | 0,824378 [68] | 0,988968 [69] | 4 |
| M4 | 30 | 0,567592 [70] | 0,826353 [71] | 0,990417 [72] | 5 |
| MA | 28 | 0,569000 [73] | 0,827668 [74] | 0,992578 [75] | 6 |
| MB | 28 | 0,568644 [76] | 0,833324 [77] | 1,022862 [78] | 7 |
| MC | 29 | 0,569601 [79] | 0,827939 [80] | 0,993240 [81] | 8 |
