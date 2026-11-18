# painel/

Tem: a aplicação, em três abas: exploração, explicação e diagnóstico. Ela só lê arquivos prontos e não treina nada. Aponta para: nenhuma pasta; é o fim do caminho.

| arquivo | o que faz | lê | grava |
|---|---|---|---|
| `app.py` | Monta as 3 abas e a barra lateral. Para antes de ler dado se o servidor não estiver em localhost ou se as estatísticas de uso estiverem ligadas. | Pelos módulos abaixo. | Nada. |
| `dados.py` | Lê os registros, a base por bairro e mês e a malha e faz as contagens da aba de exploração. | `dados/processados/eventos.csv`, `dados/processados/painel_bairro_mes_features.csv`, `dados/processados/dicionario_painel_features.csv` e `dados/geograficos/bairros_salvador.geojson`. | Nada. |
| `saidas_do_modelo.py` | Lê as explicações e as métricas. O `app.py` o importa com o nome `modelo`. | `resultados/explicacoes.csv`, `resultados/explicacoes_info.json`, `resultados/validacao_metricas.csv` e `resultados/validacao_metricas_mensais.csv`. | Nada. |
| `rotulos.py` | Define rótulos, cores, letra mínima e os textos de interface (`TEXTOS`). | Nada. | Nada. |
| `visoes_shap.py` | Desenha as 4 visões de explicação: cascata, enxame, dependência e mudança entre anos. | Nada. | Nada. |
| `.streamlit/config.toml` | Fixa o servidor em localhost, desliga as estatísticas de uso e esconde a barra de ferramentas. O Streamlit lê este arquivo por estar na pasta do `app.py`. | Nada. | Nada. |

Da raiz do repositório: `python -m streamlit run painel/app.py`. O fundo dos mapas vem de um servidor de mapas na internet. Sem internet, os mapas abrem sem o fundo.
