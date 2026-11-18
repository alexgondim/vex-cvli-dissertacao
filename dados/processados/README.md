# dados/processados/

Tem: os registros tratados, com uma linha por vítima; a base por bairro e mês, com uma linha para cada mês de cada bairro; e o dicionário das colunas da base. Aponta para: `modelo/` e `experimentos/`, que leem a base, e `painel/`, que lê os três arquivos.

| arquivo | o que é | quem grava | quem lê |
|---|---|---|---|
| `eventos.csv` | Os registros tratados. Tem 7.766 linhas, uma por vítima, com data, turno, bairro, RISP, AISP, natureza, sexo e idade da vítima. | `preprocessamento/gerar_base_bairro_mes.py` | `painel/dados.py` |
| `painel_bairro_mes_features.csv` | A base por bairro e mês. Tem 11.952 linhas, uma por bairro e mês de referência (166 bairros e 72 meses), e 35 colunas: os 30 atributos, o alvo `target_cvli` e 4 colunas de data. | `preprocessamento/gerar_base_bairro_mes.py` | `modelo/comum.py`, e por ele os programas de `modelo/` e `experimentos/`; `painel/dados.py` |
| `dicionario_painel_features.csv` | O dicionário da base. Tem 35 linhas, uma por coluna, na ordem do arquivo da base. | `preprocessamento/gerar_base_bairro_mes.py` | `modelo/comum.py` e `painel/dados.py` |

O dicionário da base tem o mesmo nome do dicionário de origem, em `dados/brutos/`. O de origem tem 32 linhas. O gerador troca `prop_noturno_hist12` pelas 3 proporções de turno (`prop_madrugada_hist12`, `prop_tarde_hist12` e `prop_noite_hist12`) e acrescenta a coluna `aisp`.
