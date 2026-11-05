"""Itens 1 e 2 do bloco 26: portão da construção e protocolo, gravado antes do primeiro ajuste na base.

O portão confere a base de 27/09 contra os números do bloco e para no primeiro que não bater. A grade é a do
XGBoost de 24/09, lida do programa congelado daquele dia. Os padrões da biblioteca saem de uma árvore de 1
rodada ajustada em dado inventado de 2 linhas: nenhuma linha da base passa por ajuste aqui.
"""
import ast
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

import comum

comum.registrar("início da rodada; etapa 1, portão da construção")
falhas = []


def confere(nome, obtido, esperado):
    ok = obtido == esperado
    comum.registrar("portão: %s = %s (esperado %s) %s" % (nome, obtido, esperado, "confere" if ok else "NÃO CONFERE"))
    if not ok:
        falhas.append(nome)


confere("sha256 da base", comum.sha256(comum.BASE), comum.SHA_BASE)
bruta = pd.read_csv(comum.BASE)
dicionario = pd.read_csv(comum.DICIONARIO)
confere("linhas", len(bruta), 11952)
confere("colunas", bruta.shape[1], 35)
confere("colunas na ordem do dicionário", bruta.columns.tolist() == dicionario["coluna"].tolist(), True)
jan19 = bruta[(bruta["ano_alvo"] == 2019) & (bruta["mes_alvo"] == 1)]
grade_2018 = int(sum(jan19["cvli_lag%d" % k].sum() for k in range(1, 13)))
confere("grade, registros de 2018 (defasagens das linhas de jan/2019) mais todos os alvos",
        grade_2018 + int(bruta[comum.ALVO].sum()), 7766)
anos = bruta["ano_alvo"]
confere("linhas com alvo de 2019 a 2022", int(anos.between(2019, 2022).sum()), 7968)
confere("linhas com alvo em 2023", int((anos == 2023).sum()), 1992)
confere("linhas com alvo em 2024", int((anos == 2024).sum()), 1992)
lista = comum.atributos()
confere("atributos", len(lista), 30)
confere("blocos (criminal, temporal, espacial, perfil)", [len(v) for v in comum.BLOCOS.values()], [16, 4, 3, 7])
if falhas:
    comum.registrar("PARADA: o portão não confere em %s" % falhas)
    sys.exit(1)

# a grade do XGBoost no programa congelado de 24/09
texto = comum.GRADE_24_09.read_text(encoding="utf-8")
achado = re.search(r'"xgb": \[dict\(learning_rate=a, n_estimators=b, max_depth=c, min_child_weight=d\)\s*'
                   r'for a, b, c, d in itertools\.product\((\[[^\]]*\]), (\[[^\]]*\]), (\[[^\]]*\]), (\[[^\]]*\])\)\]', texto)
assert achado, "grade do XGBoost não encontrada no programa de 24/09"
listas = [ast.literal_eval(g) for g in achado.groups()]
assert listas == [comum.TAXAS, comum.ARVORES, comum.PROFUNDIDADES, comum.PESOS], listas
assert len(comum.GRADE) == 36

# padrões da biblioteca, de uma árvore de 1 rodada em dado inventado
pequeno = xgb.XGBRegressor(objective="count:poisson", n_estimators=1).fit(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]))
config = json.loads(pequeno.get_booster().save_config())
arvore = config["learner"]["gradient_booster"]["tree_train_param"]
padrao_lido = dict(learning_rate=round(float(arvore["eta"]), 6), n_estimators=xgb.XGBRegressor().get_num_boosting_rounds(),
                   max_depth=int(arvore["max_depth"]), min_child_weight=float(arvore["min_child_weight"]))
assert padrao_lido == dict(comum.PADRAO, min_child_weight=1.0), padrao_lido
posicao_padrao = comum.GRADE.index(comum.PADRAO) + 1
threads = config["learner"]["generic_param"]["nthread"]
comum.registrar("grade de 24/09 lida: 36 configurações; padrão da biblioteca %s, na posição %d da grade; nthread %s, "
                "%d núcleos" % (padrao_lido, posicao_padrao, threads, os.cpu_count()))
comum.registrar("etapa 2, protocolo")

if comum.PROTOCOLO.exists():
    comum.registrar("PARADA: o protocolo já existe e não é regravado")
    sys.exit(1)
linhas_grade = "\n".join("| %d | %s | %d | %d | %d |%s" % (i + 1, str(h["learning_rate"]).replace(".", ","), h["n_estimators"],
                                                            h["max_depth"], h["min_child_weight"],
                                                            " padrão da biblioteca |" if h == comum.PADRAO else " |")
                         for i, h in enumerate(comum.GRADE))
blocos = "\n".join("- %s (%d): %s" % (nome, len(v), ", ".join("`%s`" % a for a in v)) for nome, v in comum.BLOCOS.items())
protocolo = """# Protocolo da rodada de 07/10 (bloco 26)

Gravado em {data}, antes do primeiro ajuste na base.

Base: `analises/medicoes_2026-09/base_2018_2024/painel_bairro_mes_features.csv`, sha256 `{sha}`. Portão do item 1:
confere em todos os números (log `rodada.log`).

## a. Grade do XGBoost

As 36 configurações da busca de 24/09 (`congelado_2026-09-24/hiperparametros_busca.py`, grade `xgb`), na ordem
daquele programa, sem acrescentar nem tirar nenhuma:

| ordem | learning_rate | n_estimators | max_depth | min_child_weight | |
|---|---|---|---|---|---|
{grade}

A configuração padrão da biblioteca está na grade, na posição {posicao}: eta {eta}, max_depth {prof}, min_child_weight
{peso} e {rodadas} árvores. Os três primeiros foram lidos da configuração de uma árvore de 1 rodada ajustada em dado
inventado de 2 linhas; o número de árvores, de `XGBRegressor().get_num_boosting_rounds()`. Nenhuma linha da base foi
usada nessa leitura.

## b. Critério de escolha

Menor deviance de Poisson média nas 1.992 previsões das 12 dobras de 2023, com piso de 0,01 nas estimativas (o piso
vale só na deviance). Em caso de empate até a 6ª casa decimal, fica a configuração que vem antes na grade.

Dobras: para cada mês-alvo de 2023, treino com todas as linhas de alvo anterior àquele mês, desde jan/2019, e
previsão das 166 linhas do mês.

## c. Estimador, semente, versão e threads

- `XGBRegressor(objective="count:poisson", enable_categorical=True, random_state=42)`, com os 4 hiperparâmetros da
  grade; o resto no padrão da biblioteca. Um modelo só para a cidade.
- Semente 42.
- XGBoost {versao}.
- Threads: `n_jobs` no padrão (`nthread` {threads} na configuração, isto é, todos os núcleos); a máquina tem {nucleos}.
- `bairro`, `aisp` e `risp` entram como categóricas nativas (pandas `Categorical`, com o conjunto inteiro de valores
  em todas as dobras), sem colunas indicadoras. No padrão da biblioteca, `max_cat_to_onehot` = 4: a `risp`, com 3
  categorias, é dividida uma categoria contra as outras; `bairro` (166) e `aisp` (16), por conjunto de categorias.
- Proporções e idade média ausentes continuam ausentes, sem preenchimento.

## d. Atributos, por bloco

Os 30 atributos, na ordem do dicionário (`dicionario_painel_features.csv`) em todos os modelos:

{blocos}

## e. Ordem das etapas

1. Portão da construção.
2. Este protocolo.
3. Busca dos hiperparâmetros nas 12 dobras de 2023 (36 configurações).
4. Ablação M1 a M4 com a configuração escolhida; o M4 tem de repetir a busca.
5. Sensibilidade territorial (M4, MA, MB e MC), com contribuições e tau-b de Kendall.
6. Congelamento (`CONGELADO.tsv`).
7. Modelo do corte fixo, nas 7.968 linhas de alvo de 2019 a 2022.
8. Contribuições e interações do modelo do corte fixo.
9. Tabelas dos 5 usos do SHAP.
10. Painel ligado às saídas da rodada.
11. Capturas.
12. Zip da Parte 1 e parada. Os alvos de 2024 não entram em métrica nesta parte.
""".format(data=comum.agora(), sha=comum.SHA_BASE, grade=linhas_grade, posicao=posicao_padrao,
           eta=str(padrao_lido["learning_rate"]).replace(".", ","), prof=padrao_lido["max_depth"],
           peso=int(padrao_lido["min_child_weight"]), rodadas=padrao_lido["n_estimators"], versao=xgb.__version__,
           threads=threads, nucleos=os.cpu_count(), blocos=blocos)
comum.PROTOCOLO.write_text(protocolo, encoding="utf-8")
comum.registrar("protocolo gravado: protocolo.md, sha256 %s" % comum.sha256(comum.PROTOCOLO))
