# Números do capítulo: ablação com a composição dos blocos de 21/10 (Parte 4 da rodada de 07/10)

Extraídos por `numeros_capitulo.py` de `ablacao_blocos2_2023.csv`, `ablacao_blocos2_2024.csv` e `resumo_parte4.json` desta pasta, que são cópias das saídas de `rodada_2026-10-07/parte4/` (sha256 em `HASHES.tsv`). Valores em 6 casas, arredondados da precisão plena dos arquivos. O número entre colchetes é a linha de `numeros_capitulo.csv`, que traz o arquivo e o local de cada valor.

- Deviance de Poisson média, com piso de 0,01 na previsão.
- Blocos de 21/10: criminal (16), temporal (8: os 4 do calendário e as proporções de madrugada, tarde, noite e fim de semana), espacial (3) e perfil (3: `prop_masc_hist12`, `idade_media_hist12`, `n_vitimas_hist12`). M1 criminal (16 atributos), M2 + temporal (24), M3 + espacial (27), M4 os 30.
- 2023: média das 12 dobras da janela expansiva (166 linhas por dobra). 2024: corte fixo, treino com alvos de 2019 a 2022 e avaliação nas 1.992 linhas de 2024. Configuração 3 da grade, semente 42.
- Ganho do bloco: deviance do modelo com o bloco menos a do modelo anterior; negativo é melhora.

## Deviance de M1 a M4

| modelo | atributos | 2023, média das 12 dobras | 2024, corte fixo |
|---|---|---|---|
| M1 | 16 | 1,048960 [1] | 1,024669 [5] |
| M2 | 24 | 1,046502 [2] | 1,024970 [6] |
| M3 | 27 | 1,005707 [3] | 0,991567 [7] |
| M4 | 30 | 1,007328 [4] | 0,990417 [8] |

## Ganho de cada bloco em deviance

| bloco | diferença | 2023, média das 12 dobras | 2024, corte fixo |
|---|---|---|---|
| temporal | M2 − M1 | -0,002458 [9] | 0,000301 [12] |
| espacial | M3 − M2 | -0,040795 [10] | -0,033403 [13] |
| perfil | M4 − M3 | 0,001621 [11] | -0,001150 [14] |
