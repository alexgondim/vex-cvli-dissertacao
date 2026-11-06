# VEX-CVLI: código da modelagem mensal por bairro e da ablação

Material de apoio à dissertação de mestrado de Alex Gondim Lima (PGCOMP/UFBA), orientada pelo Prof. Danilo Coimbra, preparado para a auditoria pedida pela banca de qualificação em 04/11/2026. Repositório privado.

Está aqui o que produziu as tabelas e os números do Capítulo 4: os programas da rodada de 07 a 22/10/2026, os protocolos escritos antes de cada parte, as tabelas de congelamento (sha256 dos programas e dos arquivos que cada parte fixou), os logs de execução e as saídas agregadas. Estão aqui também o gerador da base bairro-mês, os módulos que ele importa, o painel e os programas que derivam os números do capítulo a partir das saídas.

## O que não está neste repositório

- **Dados de vítimas e planilhas de origem:** `eventos.csv`, com uma linha por vítima, as planilhas de origem e a base empilhada. A base bairro-mês `painel_bairro_mes_features.csv`, que a rodada lê, está no repositório (ver "Integridade").
- **Previsões linha a linha** (`previsoes_*.csv` e `lstm_previsoes*.csv`) e `parte3/contribuicoes_MD_2024.csv`, que trazem o valor observado por bairro e mês, e `shap_b_dependencia_2024.csv`, que traz valores de atributo por linha. Os programas os regravam a partir da base.
- **`painel_mensal/dados_modelo/`**, que o painel lê: é exportada da rodada por `analises/medicoes_2026-09/rodada_2026-10-07/exportar_painel.py` e `parte2/exportar_metricas.py`.
- **Os dicionários e a base sintética de exemplo de autoria do orientador** (`analises/medicoes_2026-09/orientador/`). Sem a base sintética, o modo de demonstração do painel não abre; sem os dicionários e sem as planilhas de origem, o gerador não roda.
- **As cópias das saídas da rodada** que as pastas `congelado_2026-10-13/`, `congelado_2026-10-13b/` e `congelado_2026-10-21/` guardavam. A correspondência de cada cópia omitida com o arquivo igual deste repositório, pelo sha256, está em `copias_omitidas.tsv`; as que não têm igual aqui são as previsões linha a linha, o `parte3/contribuicoes_MD_2024.csv`, o `teste.json` exportado para `painel_mensal/dados_modelo/` e a tabela de hashes de cada pasta.

## Estrutura

```
analises/medicoes_2026-09/
  xlsx_min.py                   leitor de planilha com a biblioteca padrão, importado pelo gerador
  base_2018_2024/               a base bairro-mês, o gerador que a produz, dicionários, hashes das entradas
                                e saídas, números da seção de dados do capítulo
  congelado_2026-09-24/         grade de hiperparâmetros fixada em 24/09 (hiperparametros_busca.py),
                                lida como texto pela rodada
  congelado_2026-10-13/         numeros_capitulo.py, .md e .csv de cada congelamento: ligam as saídas
  congelado_2026-10-13b/        aos números do capítulo (o .py lê as cópias da própria pasta; ver
  congelado_2026-10-21/         copias_omitidas.tsv)
  rodada_2026-10-07/            Parte 1: busca de configuração, sensibilidade, corte fixo e tabelas SHAP;
                                ablacao_blocos2.py, da Parte 4, fica na raiz
    parte2/                     formas de validação e o teste de 2024 (corte fixo, ablação e janela),
                                com a pasta ensaio/
    parte3/                     variante MD e teste de viabilidade da LSTM
    parte4/                     ablação por blocos (M1 a M4) e comparação com a Parte 1
    parte5/                     as 8 combinações espaciais (M4 e MA a MG)
data/processed/                 tabela da Portaria SSP 435/2020 (AISP por bairro), mapeamento de grafias
                                de bairro e GeoJSON dos bairros
painel_mensal/                  painel (app.py e módulos), configuração, programas de figura e de conferência
src/utils/                      módulos que o gerador importa (AISP por bairro e normalização de naturezas)
requisitos.txt                  versões das 13 bibliotecas da rodada, do gerador, do painel e das figuras
HASHES.tsv                      sha256 de todos os arquivos deste repositório
copias_omitidas.tsv             cópias das pastas congeladas que ficaram fora, com o arquivo igual daqui
```

A tabela da Portaria 435/2020 tem 165 bairros; nomes que não estão nela, como Vista Alegre e os trechos da Avenida Paralela, recebem AISP por regra escrita em `src/utils/aisp_crosswalk.py`.

## Ambiente

A rodada foi executada em Windows, com Python 3.11.9, num ambiente virtual. Os protocolos gravaram, na hora da execução, as versões lidas por `__version__`: XGBoost 3.0.2 (`protocolo.md`, linha 70; `parte2/protocolo_parte2.md`, linha 8; `parte3/protocolo_parte3.md`, linha 8; `parte5/protocolo_parte5.md`, linha 9), NumPy 2.4.4 e SciPy 1.17.1 (os mesmos três protocolos das Partes 2, 3 e 5), pandas 3.0.2 (Parte 5, linha 9) e PyTorch 2.14.1+cpu (Parte 3, linha 8; Parte 5, linha 9). As demais versões da tabela foram lidas por importação, no mesmo ambiente, em 06/11/2026, com uma exceção: o `__version__` do plotly lê o registro de instalação, e o ambiente guarda um registro antigo, de modo que ele devolve 6.1.2; os arquivos instalados são os da 6.7.0, conferidos um a um pelo RECORD do pacote, e é essa a versão declarada. O arquivo `requisitos.txt` fixa as 13.

| pacote | versão | onde é usado |
|---|---|---|
| xgboost | 3.0.2 | rodada (estimador); demonstração do painel |
| numpy | 2.4.4 | rodada e painel |
| pandas | 3.0.2 | rodada e painel |
| scipy | 1.17.1 | rodada |
| torch | 2.14.1+cpu | Partes 3 e 5 |
| Unidecode | 1.4.0 | gerador, pela normalização de naturezas |
| streamlit | 1.61.0 | painel |
| plotly | 6.7.0 | painel |
| shap | 0.51.0 | painel |
| shapely | 2.1.1 | painel |
| matplotlib | 3.10.8 | painel |
| playwright | 1.62.0 | programas de figura |
| pillow | 12.2.0 | programas de figura |

Para recriar o ambiente: Python 3.11 e `pip install -r requisitos.txt`. O `torch==2.14.1+cpu` é a distribuição para CPU do PyTorch, servida pelo índice do próprio PyTorch (`--extra-index-url https://download.pytorch.org/whl/cpu`), e não pelo PyPI. Não há lista completa do ambiente: o `pip freeze` do ambiente virtual devolve versões antigas de 46 pacotes, porque cada um deles tem dois registros de instalação, e por isso não foi incluído.

## Como a rodada se reproduz

**A árvore não pode ser renomeada.** O `comum.py` da rodada encontra a raiz do repositório a partir da própria pasta (`PASTA = Path(__file__).resolve().parent` e `RAIZ = PASTA.parents[2]`, três níveis acima da pasta da rodada) e escreve no código os caminhos da base, do dicionário e da grade de 24/09. O gerador e os módulos de `src/utils/` fazem o mesmo. Renomear `analises/`, `medicoes_2026-09/`, `rodada_2026-10-07/`, `base_2018_2024/`, `congelado_2026-09-24/`, `painel_mensal/`, `data/` ou `src/` quebra os programas congelados.

**Os programas executam ao ser importados.** Dos 32 programas da rodada, só `ablacao_blocos2.py` tem a guarda `if __name__ == "__main__"`. Importar qualquer outro o executa, e os `comum*.py` leem arquivos ao ser importados. Cada programa roda com `python <caminho do programa> [modo]`; os caminhos de leitura e gravação são calculados a partir da localização do programa, não da pasta corrente.

**Pasta de execução.** A reprodução acontece numa cópia deste repositório, não nele: ele é a referência com a qual as saídas regeneradas são comparadas. A cópia começa só com os programas (`.py`), os protocolos (`protocolo*.md`, inclusive `parte4/protocolo_parte4.md`, que não tem programa gerador), a grade de 24/09, os arquivos de `data/processed/` e, em `analises/medicoes_2026-09/base_2018_2024/`, a base bairro-mês `painel_bairro_mes_features.csv` e o `dicionario_painel_features.csv`, como estão neste repositório. As tabelas de congelamento, as saídas e os logs ficam fora da cópia: com uma tabela de congelamento presente, os modos de ensaio e os programas de congelamento param (`parte2/comum2.py`, linha 69; `parte3/comum3.py`, linha 83; `parte5/comum5.py`, linha 69; `ablacao_blocos2.py`, linha 167). A pasta `painel_mensal/dados_modelo/` deve existir na cópia antes do passo 7, se o programa não a criar.

**Ordem.** A numeração e a ordem são as dos logs da rodada (`rodada.log` e `rodada_parte2.log` a `rodada_parte5.log`). O modo vai na linha de comando. Os passos 0, 1, 8, 19 e 30 não rodam com os arquivos presentes e ficam de fora: o gerador (0) recusa rodar com a base presente e depende das planilhas de origem; `portao_protocolo.py` (1) confere a base e a grade e para porque `protocolo.md` já existe; `protocolo_parte2.py`, `protocolo_parte3.py` e `protocolo_parte5.py` (8, 19 e 30) recusam sobrescrever o protocolo.

| passo | programa (modo) | grava |
|---|---|---|
| 2 | `busca.py` | `busca_configuracoes.csv`, `previsoes_busca.csv`, `escolha.json`, `busca_ranking.csv` |
| 3 | `ablacao_sensibilidade.py` | `ablacao.csv`, `previsoes_ablacao_sensibilidade.csv`, `contribuicoes_sensibilidade.csv` e as 4 tabelas `sensibilidade_*.csv` |
| 4 | `congelar.py` | `CONGELADO.tsv` |
| 5 | `corte_fixo.py` | `modelo_corte_fixo.json`, `estimativas_corte_fixo.csv`, `contribuicoes_corte_fixo.csv`, `interacoes_2024.csv` |
| 6 | `tabelas_shap.py` | as 6 tabelas `shap_a` a `shap_e` |
| 7 | `exportar_painel.py` | `painel_mensal/dados_modelo/explicacoes.csv` e `rodada.json` |
| 9 a 12 | `parte2/forma_corte_fixo.py`, `forma_ablacao.py`, `forma_janela.py`, `juntar_teste.py` (`ensaio`) | `parte2/ensaio/` |
| 13 | `parte2/congelar_parte2.py` | `CONGELADO_parte2.tsv` |
| 14 a 17 | os mesmos quatro (`teste`) | `parte2/` |
| 18 | `parte2/exportar_metricas.py` | cópias em `painel_mensal/dados_modelo/` e `teste.json` |
| 20 a 22 | `parte3/md_validacao.py`, `lstm_ensaio.py`, `permutacao.py` (`ensaio`) | `parte3/`, inclusive `lstm_modelo.pt` |
| 23 | `parte3/congelar_parte3.py` | `CONGELADO_parte3.tsv` |
| 24 e 25 | `parte3/md_teste.py`, `lstm_teste.py` (`teste`) | `parte3/` |
| 26 | `ablacao_blocos2.py validacao` | `parte4/ablacao_blocos2_2023.csv`, `previsoes_ablacao_blocos2_2023.csv`, `conferencia_2023.json` |
| 27 | `ablacao_blocos2.py congelar` | `parte4/CONGELADO_parte4.tsv` |
| 28 | `ablacao_blocos2.py teste` | `parte4/ablacao_blocos2_2024.csv`, `previsoes_ablacao_blocos2_2024.csv`, `resumo_parte4.json` |
| 29 | `parte4/comparacao_parte4.py` | `comparacao_partes_1_e_4.csv` |
| 31 a 33 | `parte5/validacao_2023.py`, `teste_2024.py`, `tabela_combinacoes.py` (`ensaio`) | `parte5/`; os dois últimos só registram no log |
| 34 | `parte5/congelar_parte5.py` | `CONGELADO_parte5.tsv` |
| 35 e 36 | `parte5/teste_2024.py`, `tabela_combinacoes.py` (`teste`) | `conferencia_2024.json`, `sensibilidade_parte5_2024.csv`, `tabela_8_combinacoes.csv` |

Os modos de teste das Partes 2, 3 e 5 e o `ablacao_blocos2.py` exigem a tabela de congelamento da própria parte e conferem sha256 por ela (`parte2/comum2.py`, linhas 71 a 75; `parte3/comum3.py`, a partir da linha 85; `parte5/comum5.py`, a partir da linha 71; `ablacao_blocos2.py`, linhas 222 a 243). Os programas da Parte 1 `busca.py`, `ablacao_sensibilidade.py`, `corte_fixo.py`, `tabelas_shap.py` e `exportar_painel.py` regravam as próprias saídas sem trava; os demais recusam sobrescrever.

**Cronologia das avaliações com alvo de 2024**, pelos logs e pelas tabelas de congelamento. A Parte 1 (07/10) não calculou métrica com o alvo de 2024: o modelo do corte fixo e as tabelas SHAP estimam as linhas de 2024 sem usar o alvo. A primeira avaliação com o alvo de 2024 é a Parte 2, em 13/10, das 15:08 às 15:11, com o corte fixo, a ablação na composição da Parte 1 com MA, MB e MC, e a janela expansiva, depois do protocolo (15:04) e do congelamento (15:07). As avaliações seguintes em 2024 foram definidas depois desse resultado: MD e LSTM na Parte 3, em 13/10 às 18:25; M1 a M4 na composição revista dos blocos na Parte 4, em 21/10 às 17:37; MD repetida, ME, MF, MG e a tabela das 8 combinações na Parte 5, em 22/10 às 10:19, cujo protocolo (linha 5) registra que a combinação bairro e AISP foi pedida pelo orientador ao ler a tabela de sensibilidade do capítulo, e que as outras duas completam as 8 combinações. Cada programa de teste rodou uma única vez (`rodada_parte2.log` a `rodada_parte5.log`), e a configuração do modelo foi fixada em 07/10, na validação de 2023, antes de qualquer leitura de 2024.

**Conferência.** Depois da execução, cada arquivo regenerado é comparado com o deste repositório pelo sha256 (`HASHES.tsv`), e cada tabela de congelamento nova, com a original. Diferenças esperadas:

- os logs, que registram data e hora;
- `rodada_2026-10-07/tabelas_shap.py`: o `CONGELADO.tsv` original registra `61cb3bcf…` para esse programa, e o arquivo deste repositório tem `e603e49c…`. O programa foi alterado depois do congelamento, em 07/10 às 17:59:04, para renomear colunas da tabela espacial; o `rodada.log` registra a alteração com os dois sha256, seguida da execução que gravou as tabelas `shap_*` deste repositório;
- os arquivos da LSTM (`lstm_modelo.pt` e as saídas `lstm_*` e `permutacao_*`), gerados com torch 2.14.1+cpu e semente fixada: a reprodução bit a bit em outra plataforma não foi verificada;
- `busca_configuracoes.csv` e `busca_ranking.csv`, que trazem a coluna `segundos`, com a duração de cada ajuste;
- as tabelas de congelamento, que trazem data e hora. Um `CONGELADO.tsv` regravado lista 9 programas, e não 7: `exportar_painel.py` e `ablacao_blocos2.py` entraram na raiz da rodada depois de 07/10 (o segundo está no `CONGELADO_parte4.tsv`; o primeiro, em nenhuma tabela);
- as tabelas `shap_*`, que trazem os rótulos de bloco da composição da Parte 1 (16, 4, 3 e 7; `tabelas_shap.py`, linhas 20 e 29, lê `comum.BLOCOS`), de modo que as proporções de turno e de fim de semana aparecem como `perfil`. A composição do capítulo é a da Parte 4 (16, 8, 3 e 3), em que elas são temporais. As contribuições não mudam com o rótulo.

**O que a LSTM é aqui.** `parte3/lstm_ensaio.py`, `lstm_teste.py`, `permutacao.py`, `lstm_modelo.pt` e as saídas `lstm_*` e `permutacao_*` são um teste de viabilidade pedido pelo orientador em 13/10/2026, sobre a sequência das 12 contagens mensais mais o seno e o cosseno do mês, com uma configuração fixada antes do treino, parada antecipada em 2023 e importância por permutação das 12 defasagens comparada com o M4. A LSTM não faz parte do sistema nem do painel e não é resultado do capítulo; está aqui porque foi congelada junto com a Parte 3 (`parte3/CONGELADO_parte3.tsv`).

## Onde está cada tabela e cada número do capítulo

| item do capítulo | programa | arquivo neste repositório |
|---|---|---|
| Tabela de registros por ano; alvo zero em 67,2%; 1.566 janelas vazias; 331 registros sem idade; naturezas; turnos; bairros por RISP; 13.944 células e 11.952 linhas; variância do alvo 1,973 vez a média; média do alvo no teste (0,46) | `base_2018_2024/gerar_base_2018_2024.py` | `base_2018_2024/numeros_secao_dados.csv` |
| 6 de 7.766 registros com RISP divergente; 51 registros de idade zero em 2018 | `gerar_base_2018_2024.py` | `base_2018_2024/gerar_base_2018_2024_resultado.txt`, linhas 22 e 13 |
| Tabela dos 30 atributos: nomes e ordem; composição dos blocos (16, 8, 3 e 3) | `gerar_base_2018_2024.py`; `ablacao_blocos2.py` | `base_2018_2024/dicionario_painel_features.csv`; `congelado_2026-10-21/numeros_capitulo.md` |
| Tabela de métricas de 2024, corte fixo e janela; deviance da janela em 2022 e 2023 | `parte2/juntar_teste.py`; `congelado_2026-10-13/numeros_capitulo.py` | `parte2/metricas.csv`; `congelado_2026-10-13/numeros_capitulo.md` |
| Tabela de ablação M1 a M4, 2023 e 2024 | `ablacao_blocos2.py` | `parte4/ablacao_blocos2_2023.csv`; `parte4/ablacao_blocos2_2024.csv` |
| Tabela de sensibilidade territorial, 8 variantes | `parte5/tabela_combinacoes.py` | `parte5/tabela_8_combinacoes.csv` |
| Configuração escolhida (0,05; 100; 3; 20), deviance 1,0073 e a posição 33 no ranking da busca | `busca.py` | `escolha.json`; `busca_ranking.csv`; `rodada.log` |
| Resumo global das contribuições de 2024 (bairro 0,76) | `tabelas_shap.py` | `shap_a_global_2024.csv` |
| Itapuã, junho de 2024 (estimativa 1,7; contribuições) | `tabelas_shap.py` | `shap_c_local_itapua_2024_06.csv`; as 16 vítimas, no `rodada.log` |
| Peso espacial (98,8%; 1,2%; menos de 0,01%) | `ablacao_sensibilidade.py` | `sensibilidade_peso_espacial.csv` |
| τb entre variantes (0,95; 0,98; 0,71) | `ablacao_sensibilidade.py` | `sensibilidade_tau_b.csv` |
| τb entre M4 e MD (0,72) | `parte3/md_validacao.py` | `parte3/resumo_MD_2023.json` |

Números citados no capítulo que nenhum programa deste repositório grava, com a origem de cada um:

- a dependência da proporção da madrugada (−0,007 a 0,005): os dois extremos aparecem na coluna da madrugada de `contribuicoes_corte_fixo.csv`, nas linhas de 2024; o recorte sem janela vazia exige os valores dos atributos, que estão em `shap_b_dependencia_2024.csv`, regravado por `tabelas_shap.py` e fora deste repositório;
- a contribuição da idade média nas linhas sem idade (−0,18) e nas demais (0,04): derivável de `contribuicoes_corte_fixo.csv` com os valores da base;
- a escala própria das contribuições (bairro de −1,56 a 1,31; os outros nove de −0,29 a 0,13): conferível nas colunas de contribuição de `contribuicoes_corte_fixo.csv`, nas linhas de 2024;
- a média das estimativas do modelo em 2024 (0,57): conferível na coluna `estimativa` de `contribuicoes_corte_fixo.csv`, nas 1.992 linhas de 2024; a da média móvel (0,49): derivável de `parte2/previsoes_teste.csv`, que `parte2/juntar_teste.py` regrava e que fica fora deste repositório;
- os registros por RISP (3.854, 1.513 e 2.399): o painel os calcula de `eventos.csv` na aba de exploração;
- o total de 54 registros de idade zero e a maior contagem mensal de um bairro (13): contados na base.

As figuras do capítulo vêm de `painel_mensal/figuras_mapa.py` (mapas de contagem por bairro e de RISP e AISP) e de `painel_mensal/capturas_capitulo.py` (capturas das abas), com o painel em execução; o diagrama das abas e a figura dos registros sem preenchimento foram montados fora dos programas deste repositório.

## Integridade

`HASHES.tsv` traz o sha256 de cada arquivo. A base bairro-mês, `analises/medicoes_2026-09/base_2018_2024/painel_bairro_mes_features.csv` (sha256 `2a8288006c2323fa2f4995b82afa7e079f1944db804a0091c6ffd01224101443`, o mesmo registrado no `comum.py` da rodada e no `HASHES.tsv` da pasta), tem 11.952 linhas e 35 colunas, todas agregadas por bairro e mês; nenhuma linha é uma pessoa. É dado agregado de vítimas da SSP-BA: o repositório é privado por causa dela, não deve ser redistribuído e não pode ser tornado público com ela no histórico. O `.gitattributes` desliga a conversão de fim de linha para todos os arquivos, para que qualquer clone tenha os mesmos bytes que a máquina que executou a rodada, onde LF e CRLF convivem: os programas da rodada e do painel são LF; o gerador, os módulos de `src/utils/`, as duas tabelas CSV de `data/processed/`, as tabelas de congelamento, os logs e a maior parte das saídas e dos protocolos são CRLF (`git ls-files --eol` mostra o fim de linha de cada arquivo). Os hashes registrados nos protocolos, nas tabelas de congelamento e no `HASHES.tsv` da base são os desses bytes. Cada tabela de congelamento cobre os arquivos da própria parte; a cadeia inteira é conferida pelo `HASHES.tsv`.
