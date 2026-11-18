# resultados/

Tem: o que os programas de `modelo/` e de `experimentos/` gravam: a configuração escolhida, o modelo treinado, as contribuições, as métricas e as tabelas dos experimentos. Aponta para: `painel/`, que lê as explicações e as métricas.

As métricas são MAE, RMSE e deviance de Poisson média. As contribuições estão na escala do log da contagem esperada. "Os 5 métodos" são o modelo e as quatro referências de comparação.

| arquivo | o que é | quem grava | quem lê |
|---|---|---|---|
| `escolha.json` | A configuração escolhida na busca: ordem na grade, hiperparâmetros e as 3 métricas nas 12 dobras de 2023. | `modelo/busca_hiperparametros.py` | `modelo/treinar_modelo.py`, `modelo/exportar_para_painel.py`, `experimentos/sensibilidade_dobras_2023.py` e `experimentos/validacao_comum.py`, e por ele os outros programas de `experimentos/` |
| `modelo_treinado.json` | O modelo XGBoost treinado com os alvos de 2019 a 2022, no formato JSON do XGBoost. | `modelo/treinar_modelo.py` | Nenhum programa. É o modelo que gerou as contribuições. |
| `contribuicoes_corte_fixo.csv` | As contribuições dos 30 atributos, o valor de base e a estimativa do modelo nas 11.952 linhas da base por bairro e mês. | `modelo/treinar_modelo.py` | `modelo/exportar_para_painel.py`, `experimentos/validacao_corte_fixo.py` e `experimentos/sensibilidade_md_corte_fixo.py` |
| `explicacoes.csv` | As 11.952 linhas que o painel explica: bairro, mês-alvo, divisão, RISP, valor de base, margem, estimativa, os 30 valores e as 30 contribuições. | `modelo/exportar_para_painel.py` | `painel/saidas_do_modelo.py` |
| `explicacoes_info.json` | A identificação das explicações: modelo, períodos de treino, validação e teste, configuração, ordem na grade, atributos e escala. | `modelo/exportar_para_painel.py` | `painel/saidas_do_modelo.py` |
| `validacao_metricas_corte_fixo.csv` | As 3 métricas dos 5 métodos no corte fixo, em 2024. | `experimentos/validacao_corte_fixo.py` | `experimentos/validacao_juntar_metricas.py` |
| `validacao_previsoes_corte_fixo.csv` | As previsões dos 5 métodos no corte fixo, linha a linha, nas 1.992 linhas de 2024. | `experimentos/validacao_corte_fixo.py` | `experimentos/validacao_juntar_metricas.py` |
| `validacao_nb_corte_fixo.json` | O ajuste da binomial negativa no corte fixo: convergência, iterações, gradiente e parâmetros. | `experimentos/validacao_corte_fixo.py` | `experimentos/validacao_juntar_metricas.py` |
| `validacao_metricas_mensais.csv` | As 3 métricas dos 5 métodos em cada mês da janela expansiva, de jan/2022 a dez/2024: 180 linhas. | `experimentos/validacao_janela_expansiva.py` | `painel/saidas_do_modelo.py` e `experimentos/validacao_juntar_metricas.py` |
| `validacao_metricas_janela.csv` | As 3 métricas dos 5 métodos por ano na janela expansiva, em 2022, 2023 e 2024. | `experimentos/validacao_janela_expansiva.py` | `experimentos/validacao_juntar_metricas.py` |
| `validacao_previsoes_janela.csv` | As previsões dos 5 métodos na janela expansiva, linha a linha. | `experimentos/validacao_janela_expansiva.py` | `experimentos/validacao_juntar_metricas.py` |
| `validacao_nb_janela.json` | Os 36 ajustes mensais da binomial negativa na janela expansiva. | `experimentos/validacao_janela_expansiva.py` | `experimentos/validacao_juntar_metricas.py` |
| `validacao_metricas.csv` | As 20 linhas de métricas: corte fixo de 2024 e janela expansiva em 2022, 2023 e 2024, para os 5 métodos. | `experimentos/validacao_juntar_metricas.py` | `painel/saidas_do_modelo.py` |
| `ablacao_blocos_2023.csv` | M1 a M4 nas 12 dobras de 2023: métricas por dobra, a média das 12 dobras e as 1.992 linhas juntas. | `experimentos/ablacao_blocos.py`, modo `validacao` | O próprio `experimentos/ablacao_blocos.py`, no modo `teste`. |
| `ablacao_blocos_2024.csv` | M1 a M4 no corte fixo, com avaliação em 2024. | `experimentos/ablacao_blocos.py`, modo `teste` | Nenhum programa. É tabela final. |
| `sensibilidade_metricas.csv` | As 3 métricas de M4, MA, MB e MC nas 12 dobras de 2023. | `experimentos/sensibilidade_dobras_2023.py` | `experimentos/sensibilidade_tabela_8_combinacoes.py` |
| `sensibilidade_contribuicoes.csv` | As contribuições de M4, MA, MB e MC nas 1.992 linhas de 2023, cada linha pelo modelo da dobra dela. | `experimentos/sensibilidade_dobras_2023.py` | `experimentos/sensibilidade_md_dobras_2023.py` e `experimentos/sensibilidade_tabela_8_combinacoes.py` |
| `sensibilidade_importancia.csv` | A média do valor absoluto da contribuição dos 27 atributos não espaciais e a ordenação deles, em M4, MA, MB e MC, em 2023. | `experimentos/sensibilidade_dobras_2023.py` | `experimentos/sensibilidade_md_dobras_2023.py` e `experimentos/sensibilidade_variantes_dobras_2023.py` |
| `sensibilidade_tau_b.csv` | O tau-b de Kendall entre as ordenações dos 27 atributos não espaciais, nos 6 pares de M4, MA, MB e MC. | `experimentos/sensibilidade_dobras_2023.py` | `experimentos/sensibilidade_tabela_8_combinacoes.py` |
| `sensibilidade_corte_fixo_2024.csv` | As 3 métricas de M1 a M4, MA, MB e MC no corte fixo, em 2024. | `experimentos/sensibilidade_corte_fixo.py` | `experimentos/sensibilidade_tabela_8_combinacoes.py` |
| `sensibilidade_md_dobras_2023.csv` | As 3 métricas da MD nas 12 dobras de 2023 e no corte fixo de 2023. | `experimentos/sensibilidade_md_dobras_2023.py` | `experimentos/sensibilidade_tabela_8_combinacoes.py` |
| `sensibilidade_md_contribuicoes_2023.csv` | As contribuições da MD nas 1.992 linhas de 2023. | `experimentos/sensibilidade_md_dobras_2023.py` | `experimentos/sensibilidade_tabela_8_combinacoes.py` |
| `sensibilidade_md_resumo_2023.json` | O resumo da MD em 2023: métricas, ordenações, médias do valor absoluto, tau-b e peso espacial. | `experimentos/sensibilidade_md_dobras_2023.py` | `experimentos/sensibilidade_md_corte_fixo.py` e `experimentos/sensibilidade_tabela_8_combinacoes.py` |
| `sensibilidade_md_corte_fixo_2024.csv` | As 3 métricas da MD no corte fixo, em 2024. | `experimentos/sensibilidade_md_corte_fixo.py` | `experimentos/sensibilidade_tabela_8_combinacoes.py` |
| `sensibilidade_variantes_dobras_2023.csv` | As 3 métricas de ME, MF e MG nas 12 dobras de 2023. | `experimentos/sensibilidade_variantes_dobras_2023.py` | `experimentos/sensibilidade_tabela_8_combinacoes.py` |
| `sensibilidade_mf_contribuicoes_2023.csv` | As contribuições da MF nas 1.992 linhas de 2023. | `experimentos/sensibilidade_variantes_dobras_2023.py` | `experimentos/sensibilidade_tabela_8_combinacoes.py` |
| `sensibilidade_variantes_resumo_2023.json` | O resumo de ME, MF e MG em 2023: tau-b contra o M4, peso espacial e atributo de maior média. | `experimentos/sensibilidade_variantes_dobras_2023.py` | `experimentos/sensibilidade_tabela_8_combinacoes.py` |
| `sensibilidade_variantes_corte_fixo_2024.csv` | As 3 métricas de ME, MF e MG no corte fixo, em 2024. | `experimentos/sensibilidade_variantes_corte_fixo.py` | `experimentos/sensibilidade_tabela_8_combinacoes.py` |
| `sensibilidade_tabela_8_combinacoes.csv` | A tabela das 8 combinações de atributos espaciais, com a deviance em 2023 e em 2024, o peso espacial, o atributo de maior média e o tau-b contra o M4. | `experimentos/sensibilidade_tabela_8_combinacoes.py` | Nenhum programa. É tabela final. |

Ao rodar, os programas gravam também arquivos de apoio, como previsões linha a linha e resumos intermediários, que não fazem parte do repositório.
