"""Partes comuns do modelo: leitura da base por bairro e mês, atributos por bloco, grade de hiperparâmetros,
estimador, métricas, as 12 dobras de 2023 e as contribuições.

Lê `dados/processados/painel_bairro_mes_features.csv` e `dados/processados/dicionario_painel_features.csv`. Não grava
arquivo: os programas que o importam gravam em `resultados/` (a constante `PASTA`).

É importado pelos programas de `modelo/` e, pelo `experimentos/validacao_comum.py`, pelos de `experimentos/`. Os
programas rodam da raiz do repositório, por exemplo: python modelo/busca_hiperparametros.py
"""
import datetime
import hashlib
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

RAIZ = Path(__file__).resolve().parents[1]
PASTA = RAIZ / "resultados"
PASTA_BASE = RAIZ / "dados" / "processados"
BASE = PASTA_BASE / "painel_bairro_mes_features.csv"
DICIONARIO = PASTA_BASE / "dicionario_painel_features.csv"
ESCOLHA = PASTA / "escolha.json"

ALVO = "target_cvli"
FORA_DO_MODELO = ["ano_ref", "mes_ref", "ano_alvo", "mes_alvo", ALVO]
BLOCOS = {
    "criminal": ["cvli_lag%d" % k for k in range(1, 13)] + ["cvli_ma3", "cvli_ma6", "cvli_ma12", "cvli_trend"],
    "temporal": ["month_sin_alvo", "month_cos_alvo", "year_norm_alvo", "time_index"],
    "espacial": ["bairro", "aisp", "risp"],
    "perfil": ["n_vitimas_hist12", "prop_madrugada_hist12", "prop_tarde_hist12", "prop_noite_hist12",
               "prop_fds_hist12", "prop_masc_hist12", "idade_media_hist12"],
}
CATEGORICAS = BLOCOS["espacial"]
SEMENTE = 42
PISO = 0.01  # só na deviance
CASAS_DO_EMPATE = 6
# a grade de busca, na ordem do itertools.product: taxa, árvores, profundidade, peso mínimo
TAXAS, ARVORES, PROFUNDIDADES, PESOS = [0.05, 0.1, 0.3], [100, 300], [3, 6], [1, 5, 20]
GRADE = [dict(learning_rate=a, n_estimators=b, max_depth=c, min_child_weight=d)
         for a, b, c, d in itertools.product(TAXAS, ARVORES, PROFUNDIDADES, PESOS)]
PADRAO = dict(learning_rate=0.3, n_estimators=100, max_depth=6, min_child_weight=1)
DIVISAO = {2019: "treino", 2020: "treino", 2021: "treino", 2022: "treino", 2023: "validação", 2024: "teste"}


def agora():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def registrar(texto):
    linha = "%s  %s" % (agora(), texto)
    print(linha, flush=True)


def sha256(caminho):
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def atributos():
    """Os 30 atributos na ordem do dicionário: as colunas dele menos as de identificação e o alvo."""
    lista = [c for c in pd.read_csv(DICIONARIO)["coluna"] if c not in FORA_DO_MODELO]
    assert sorted(lista) == sorted(sum(BLOCOS.values(), [])), lista
    return lista


def em_ordem(nomes):
    """Subconjunto de atributos na ordem do dicionário."""
    return [c for c in atributos() if c in set(nomes)]


def ler_base():
    """Base por bairro e mês. Bairro, AISP e RISP viram categorias com o conjunto inteiro de valores, igual em
    todas as dobras; proporções e idade ausentes continuam ausentes."""
    b = pd.read_csv(BASE)
    for c in CATEGORICAS:
        b[c] = pd.Categorical(b[c], categories=sorted(b[c].unique()))
    b["periodo_alvo"] = pd.PeriodIndex.from_fields(year=b["ano_alvo"], month=b["mes_alvo"], freq="M")
    b["divisao"] = b["ano_alvo"].map(DIVISAO)
    return b


def estimador(config):
    return xgb.XGBRegressor(objective="count:poisson", enable_categorical=True, random_state=SEMENTE, **config)


def deviance_media(y, q):
    q = np.clip(np.asarray(q, dtype=float), PISO, None)
    y = np.asarray(y, dtype=float)
    termo = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / q), 0.0)
    return float(np.mean(2 * (termo - (y - q))))


def metricas(y, q):
    y, q = np.asarray(y, dtype=float), np.asarray(q, dtype=float)
    return {"mae": float(np.mean(np.abs(y - q))), "rmse": float(np.sqrt(np.mean((y - q) ** 2))),
            "deviance": deviance_media(y, q)}


def dobras_2023(base):
    """12 dobras: para cada mês-alvo de 2023, treino com todas as linhas de alvo anterior (desde jan/2019) e
    teste nas 166 linhas do mês."""
    saida = []
    for mes in range(1, 13):
        m = pd.Period(year=2023, month=mes, freq="M")
        saida.append((m, base["periodo_alvo"] < m, base["periodo_alvo"] == m))
    return saida


def contribuicoes(modelo, X):
    """pred_contribs na escala da margem (log da contagem esperada); a última coluna é o valor de base."""
    return modelo.get_booster().predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)


def prever_dobras(base, nomes, config, com_contribuicoes=False):
    """Previsões das 12 dobras de 2023 com os atributos dados; cada linha explicada pelo modelo da dobra dela."""
    nomes = em_ordem(nomes)
    partes, contribs = [], []
    for m, treino, teste in dobras_2023(base):
        modelo = estimador(config).fit(base.loc[treino, nomes], base.loc[treino, ALVO].to_numpy(float))
        X = base.loc[teste, nomes]
        p = base.loc[teste, ["bairro", "ano_alvo", "mes_alvo", ALVO]].copy()
        p["previsao"] = modelo.predict(X)
        partes.append(p)
        if com_contribuicoes:
            c = contribuicoes(modelo, X)
            contribs.append(pd.DataFrame(c, columns=nomes + ["valor_base"], index=X.index))
    previsoes = pd.concat(partes)
    assert len(previsoes) == 1992
    return previsoes, (pd.concat(contribs) if com_contribuicoes else None)
