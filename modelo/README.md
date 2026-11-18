# modelo/

Tem: os programas que definem o modelo, escolhem os hiperparâmetros, treinam e calculam a contribuição de cada atributo. Aponta para: `resultados/`.

| arquivo | o que faz | lê | grava |
|---|---|---|---|
| `comum.py` | Define as partes comuns: leitura da base, atributos por bloco, grade de hiperparâmetros, estimador XGBoost com perda de Poisson, métricas, as 12 dobras de 2023 e as contribuições. É importado pelos outros programas de `modelo/` e, pelo `experimentos/validacao_comum.py`, pelos de `experimentos/`. | `dados/processados/painel_bairro_mes_features.csv` e `dados/processados/dicionario_painel_features.csv`. | Nada. |
| `busca_hiperparametros.py` | Testa as 36 configurações da grade nas 12 dobras de 2023, com os 30 atributos. Escolhe a de menor deviance de Poisson. | A base por bairro e mês. | `resultados/escolha.json`; de apoio, `busca_configuracoes.csv`, `busca_ranking.csv` e `previsoes_busca.csv`. |
| `treinar_modelo.py` | Treina o modelo uma vez, com a configuração escolhida, nas 7.968 linhas de alvo de 2019 a 2022. Calcula as contribuições nas 11.952 linhas e imprime a maior diferença da aditividade. | A base por bairro e mês e `resultados/escolha.json`. | `resultados/modelo_treinado.json` e `resultados/contribuicoes_corte_fixo.csv`; de apoio, `estimativas_corte_fixo.csv` e `interacoes_2024.csv`. |
| `exportar_para_painel.py` | Prepara as explicações que o painel lê. | A base por bairro e mês, `resultados/escolha.json` e `resultados/contribuicoes_corte_fixo.csv`. | `resultados/explicacoes.csv` e `resultados/explicacoes_info.json`. |

Da raiz do repositório, nesta ordem: `python modelo/busca_hiperparametros.py`, `python modelo/treinar_modelo.py` e `python modelo/exportar_para_painel.py`. A busca retoma do que já gravou.
