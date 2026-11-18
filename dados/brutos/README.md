# dados/brutos/

Tem: a planilha de origem dos registros de CVLI de Salvador, de 2018 a 2024, como foi recebida, com uma linha por vítima, e o dicionário com a descrição de cada coluna da base. Aponta para: `preprocessamento/`.

| arquivo | o que é | quem grava | quem lê |
|---|---|---|---|
| `CVLI.xlsx` | A planilha de origem. Tem a aba `CVLI`, com 7.766 linhas, uma por vítima, e 15 colunas. | Nenhum programa: é o arquivo recebido. | `preprocessamento/gerar_base_bairro_mes.py`, pelo `preprocessamento/ler_planilha.py`. |
| `dicionario_painel_features.csv` | O dicionário de origem da base. Tem 32 linhas, uma por coluna, com grupo, descrição, fórmula e disponibilidade no momento da previsão. | Nenhum programa: é o arquivo recebido. | `preprocessamento/gerar_base_bairro_mes.py`, que o completa e grava o dicionário da base em `dados/processados/`, com o mesmo nome. |

Colunas da planilha. O pré-processamento usa 10 das 15 colunas: `QTD`, `DATA`, `ANO`, `MES`, `HORA`, `DELITO`, `BAIRRO_C`, `SEXO`, `IDADE` e `RISP 2025`. A `RISP 2025` serve só para conferir a RISP de cada registro. O dia da semana sai de `DATA`. As outras 5 colunas, `LOGRADOURO`, `LONGITUDE`, `LATITUDE`, `DIA` e `DIA DA SEMANA`, não são lidas por nenhum programa: o gerador carrega a aba inteira, mas nenhuma conta e nenhum arquivo gravado usa essas colunas.
