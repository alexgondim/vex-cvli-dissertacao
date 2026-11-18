# VEX-CVLI

Sistema preditivo e explicativo para a análise mensal de Crimes Violentos Letais Intencionais (CVLI) por bairro de Salvador, com registros de 2018 a 2024.

Um modelo XGBoost com perda de Poisson estima, para cada bairro, a contagem de registros do mês seguinte. O SHAP decompõe cada estimativa na contribuição de cada um dos 30 atributos. Um painel em três abas mostra os registros (exploração), as explicações (explicação) e o erro do modelo ao lado de quatro referências de comparação (diagnóstico).

Este repositório é privado. Ele contém registros individuais de vítimas e não deve ser redistribuído.

## Como instalar

O código roda em Python 3.11. Da raiz do repositório:

```
python -m pip install -r requisitos.txt
```

## Como abrir o painel

Da raiz do repositório:

```
python -m streamlit run painel/app.py
```

O painel só lê arquivos prontos: abre sem treinar nada. O `painel/.streamlit/config.toml` fixa o servidor em localhost e desliga as estatísticas de uso. O fundo dos mapas vem de um servidor de mapas na internet. Sem internet, os mapas abrem sem o fundo.

## Como refazer tudo a partir da planilha

Os comandos rodam da raiz do repositório, nesta ordem. Cada programa grava por cima do que encontra. O tempo é o de uma execução completa num computador de mesa e muda de uma máquina para outra.

| ordem | comando | tempo |
|---|---|---|
| 1 | `python preprocessamento/gerar_base_bairro_mes.py` | 31 s |
| 2 | `python modelo/busca_hiperparametros.py` | 137 s |
| 3 | `python experimentos/sensibilidade_dobras_2023.py` | 11 s |
| 4 | `python modelo/treinar_modelo.py` | 4 s |
| 5 | `python modelo/exportar_para_painel.py` | 3 s |
| 6 | `python experimentos/validacao_corte_fixo.py teste` | 7 s |
| 7 | `python experimentos/validacao_janela_expansiva.py teste` | 26 s |
| 8 | `python experimentos/sensibilidade_corte_fixo.py teste` | 2 s |
| 9 | `python experimentos/validacao_juntar_metricas.py teste` | 2 s |
| 10 | `python experimentos/sensibilidade_md_dobras_2023.py ensaio` | 3 s |
| 11 | `python experimentos/sensibilidade_md_corte_fixo.py teste` | 2 s |
| 12 | `python experimentos/ablacao_blocos.py validacao` | 7 s |
| 13 | `python experimentos/ablacao_blocos.py teste` | 3 s |
| 14 | `python experimentos/sensibilidade_variantes_dobras_2023.py ensaio` | 7 s |
| 15 | `python experimentos/sensibilidade_variantes_corte_fixo.py teste` | 3 s |
| 16 | `python experimentos/sensibilidade_tabela_8_combinacoes.py teste` | 2 s |

A busca grava cada configuração ao terminar. Se for interrompida, ela retoma do que já gravou.

## Mapa das pastas

| pasta | o que tem | aponta para |
|---|---|---|
| `dados/brutos/` | A planilha de origem dos registros, como foi recebida, e o dicionário com a descrição de cada coluna da base. | `preprocessamento/` |
| `dados/geograficos/` | A malha dos bairros de Salvador e as tabelas que ligam cada bairro à sua AISP e à sua RISP. | `preprocessamento/` e `painel/` |
| `preprocessamento/` | Os programas que leem a planilha, tratam os registros e montam a base por bairro e mês. | `dados/processados/` |
| `dados/processados/` | Os registros tratados, a base por bairro e mês e o dicionário das colunas da base. | `modelo/`, `experimentos/` e `painel/` |
| `modelo/` | Os programas que definem o modelo, escolhem os hiperparâmetros, treinam e calculam as contribuições. | `resultados/` |
| `experimentos/` | Os programas que avaliam o modelo: validação, ablação por blocos e sensibilidade aos atributos de território. | `resultados/` |
| `resultados/` | A configuração escolhida, o modelo treinado, as contribuições, as métricas e as tabelas dos experimentos. | `painel/` |
| `painel/` | A aplicação em três abas: exploração, explicação e diagnóstico. | nenhuma pasta |

Cada uma dessas pastas tem um `README.md` com a tabela dos seus arquivos. Na raiz, o `.gitattributes` desliga a conversão de fim de linha, e o `.gitignore` deixa fora os arquivos de apoio que os programas gravam ao rodar.

## Caminho do dado

Na tabela, "a base" é `dados/processados/painel_bairro_mes_features.csv` com `dados/processados/dicionario_painel_features.csv`, lidos pelo `modelo/comum.py`. Os arquivos sem pasta ficam em `resultados/`. "Apoio" são arquivos que o programa grava e que não fazem parte do repositório.

| entra | programa | sai |
|---|---|---|
| `dados/brutos/CVLI.xlsx`, `dados/brutos/dicionario_painel_features.csv`, `dados/geograficos/risp_por_bairro.csv`, `dados/geograficos/aisp_por_bairro.csv`, `dados/geograficos/mapeamento_bairros.csv` | `preprocessamento/gerar_base_bairro_mes.py` | `dados/processados/eventos.csv`, `dados/processados/painel_bairro_mes_features.csv`, `dados/processados/dicionario_painel_features.csv` |
| a base | `modelo/busca_hiperparametros.py` | `escolha.json`; apoio |
| a base, `escolha.json` | `experimentos/sensibilidade_dobras_2023.py` | `sensibilidade_metricas.csv`, `sensibilidade_contribuicoes.csv`, `sensibilidade_importancia.csv`, `sensibilidade_tau_b.csv`; apoio |
| a base, `escolha.json` | `modelo/treinar_modelo.py` | `modelo_treinado.json`, `contribuicoes_corte_fixo.csv`; apoio |
| a base, `escolha.json`, `contribuicoes_corte_fixo.csv` | `modelo/exportar_para_painel.py` | `explicacoes.csv`, `explicacoes_info.json` |
| a base, `escolha.json`, `contribuicoes_corte_fixo.csv` | `experimentos/validacao_corte_fixo.py` | `validacao_metricas_corte_fixo.csv`, `validacao_previsoes_corte_fixo.csv`, `validacao_nb_corte_fixo.json` |
| a base, `escolha.json` | `experimentos/validacao_janela_expansiva.py` | `validacao_metricas_mensais.csv`, `validacao_metricas_janela.csv`, `validacao_previsoes_janela.csv`, `validacao_nb_janela.json` |
| a base, `escolha.json` | `experimentos/sensibilidade_corte_fixo.py` | `sensibilidade_corte_fixo_2024.csv`; apoio |
| os 7 arquivos dos 2 programas de validação | `experimentos/validacao_juntar_metricas.py` | `validacao_metricas.csv`; apoio |
| a base, `escolha.json`, `sensibilidade_importancia.csv`, `sensibilidade_contribuicoes.csv` | `experimentos/sensibilidade_md_dobras_2023.py` | `sensibilidade_md_dobras_2023.csv`, `sensibilidade_md_contribuicoes_2023.csv`, `sensibilidade_md_resumo_2023.json`; apoio |
| a base, `escolha.json`, `contribuicoes_corte_fixo.csv`, `sensibilidade_md_resumo_2023.json` | `experimentos/sensibilidade_md_corte_fixo.py` | `sensibilidade_md_corte_fixo_2024.csv`; apoio |
| a base, `escolha.json` | `experimentos/ablacao_blocos.py` | `ablacao_blocos_2023.csv` (modo `validacao`), `ablacao_blocos_2024.csv` (modo `teste`); apoio |
| a base, `escolha.json`, `sensibilidade_importancia.csv` | `experimentos/sensibilidade_variantes_dobras_2023.py` | `sensibilidade_variantes_dobras_2023.csv`, `sensibilidade_mf_contribuicoes_2023.csv`, `sensibilidade_variantes_resumo_2023.json`; apoio |
| a base, `escolha.json` | `experimentos/sensibilidade_variantes_corte_fixo.py` | `sensibilidade_variantes_corte_fixo_2024.csv` |
| 12 arquivos dos experimentos de sensibilidade | `experimentos/sensibilidade_tabela_8_combinacoes.py` | `sensibilidade_tabela_8_combinacoes.csv` |
| `dados/processados/` (os 3 arquivos), `dados/geograficos/bairros_salvador.geojson`, `explicacoes.csv`, `explicacoes_info.json`, `validacao_metricas.csv`, `validacao_metricas_mensais.csv` | `painel/app.py` | a tela do painel |
