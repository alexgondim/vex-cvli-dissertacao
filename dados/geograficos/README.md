# dados/geograficos/

Tem: a malha dos bairros de Salvador e as tabelas que ligam cada bairro à sua AISP e à sua RISP. Aponta para: `preprocessamento/`, que usa as tabelas, e `painel/`, que desenha os mapas com a malha.

| arquivo | o que é | quem grava | quem lê |
|---|---|---|---|
| `bairros_salvador.geojson` | A malha dos bairros de Salvador. Tem 170 polígonos, com o nome do bairro em `NOME_BAIRR`. | Nenhum programa deste repositório: a malha foi montada no ArcGIS, a partir das coordenadas dos limites de bairro publicadas pela Prefeitura Municipal de Salvador. | `painel/dados.py`. |
| `aisp_por_bairro.csv` | A tabela da Portaria SSP 435/2020, Anexo I. Tem 165 linhas, com as colunas `AISP`, `nome_aisp` e `bairro`. | Nenhum programa deste repositório: é a transcrição do Anexo I da Portaria. | `preprocessamento/aisp_por_bairro.py`, importado pelo `preprocessamento/gerar_base_bairro_mes.py`. |
| `mapeamento_bairros.csv` | A grafia de cada bairro na malha. Tem 163 linhas: `bairro_principal` é o nome do bairro nos registros, e `bairro_socio` é a grafia da malha. | Nenhum programa deste repositório. | `preprocessamento/gerar_base_bairro_mes.py`. Bairro sem linha nesta tabela fica com a grafia da planilha, em caixa de título. |
| `risp_por_bairro.csv` | A RISP de cada bairro. Tem 168 linhas, com as colunas `bairro` e `risp`, e 3 RISP: ATLÂNTICO, BTS e CENTRAL. | Nenhum programa deste repositório: vem das colunas `BAIRRO_P` e `RISP` da planilha de população recebida. | `preprocessamento/gerar_base_bairro_mes.py`. |
