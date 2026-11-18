# preprocessamento/

Tem: os programas que leem a planilha, tratam os registros e montam a base por bairro e mês. Aponta para: `dados/processados/`.

| arquivo | o que faz | lê | grava |
|---|---|---|---|
| `gerar_base_bairro_mes.py` | Trata os registros e monta a base por bairro e mês, de 2018 a 2024. Completa o dicionário da base. Confere os arquivos gravados e imprime os números da base. | `dados/brutos/CVLI.xlsx`, `dados/brutos/dicionario_painel_features.csv`, `dados/geograficos/risp_por_bairro.csv`, `dados/geograficos/aisp_por_bairro.csv` e `dados/geograficos/mapeamento_bairros.csv`. | `dados/processados/eventos.csv`, `dados/processados/painel_bairro_mes_features.csv` e `dados/processados/dicionario_painel_features.csv`. |
| `ler_planilha.py` | Lê planilha `.xlsx` com a biblioteca padrão. É importado pelo `gerar_base_bairro_mes.py`. | A planilha que o gerador pede. | Nada. |
| `aisp_por_bairro.py` | Liga cada bairro à sua AISP, pela Portaria SSP 435/2020, Anexo I. Cinco nomes que a tabela não traz recebem a AISP por citação direta da Portaria, escrita no próprio programa, e um erro de digitação dos registros é corrigido. Dos cinco nomes, quatro vêm do Anexo II da Portaria 435/2020, e um, da Portaria SSP 002/2021, que a retifica. É importado pelo `gerar_base_bairro_mes.py`. | `dados/geograficos/aisp_por_bairro.csv`. | Nada. |
| `normalizar_natureza.py` | Normaliza o texto da coluna `DELITO` antes de classificar a natureza. É importado pelo `gerar_base_bairro_mes.py`. | Nada. | Nada. |

Da raiz do repositório: `python preprocessamento/gerar_base_bairro_mes.py`.
