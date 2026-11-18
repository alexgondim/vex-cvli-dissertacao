"""Monta os registros tratados e a base por bairro e mês de CVLI em Salvador, de 2018 a 2024, a partir da
planilha de origem.

Lê:
- `dados/brutos/CVLI.xlsx`, aba CVLI: as colunas QTD, DATA, ANO, MES, HORA, DELITO, BAIRRO_C, SEXO e IDADE, e a
  RISP 2025, só para conferir a RISP de cada registro;
- `dados/geograficos/risp_por_bairro.csv`: a RISP de cada bairro;
- `dados/geograficos/aisp_por_bairro.csv`, pelo `aisp_por_bairro.py`: a AISP de cada bairro, pela Portaria SSP
  435/2020, Anexo I;
- `dados/geograficos/mapeamento_bairros.csv`: a grafia de cada bairro na malha; sem malha, vale a grafia da planilha
  em caixa de título;
- `dados/brutos/dicionario_painel_features.csv`: a descrição e a fórmula de cada coluna da base.

A base tem 166 bairros x 84 meses (2018-01 a 2024-12); o alvo é a contagem em t+1; são 30 atributos; o turno entra
em três proporções, com a manhã como referência; a partição é pelo ano do alvo: treino 2019 a 2022, validação 2023,
teste 2024. Em 2018 a planilha grava idade ausente como 0, e o programa a trata como ausente.

Grava em `dados/processados/`: `eventos.csv` (uma linha por vítima), `painel_bairro_mes_features.csv` (uma linha por
bairro e mês) e `dicionario_painel_features.csv`. Confere os arquivos gravados (alvo, recontagem, proporções da
janela) e imprime as conferências e os números da base.

Da raiz do repositório: python preprocessamento/gerar_base_bairro_mes.py
"""
import hashlib
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RAIZ = Path(__file__).resolve().parents[1]
AQUI = RAIZ / "dados" / "processados"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ler_planilha import ler_tabela                                        # noqa: E402
from aisp_por_bairro import norm, CRIME_TYPO_FIX, mapear_bairros_para_aisp  # noqa: E402
from normalizar_natureza import normalize_delito                           # noqa: E402

BRUTOS = RAIZ / "dados" / "brutos"
GEO = RAIZ / "dados" / "geograficos"
ENTRADAS = {
    "CVLI.xlsx": (BRUTOS / "CVLI.xlsx", None),
    "risp_por_bairro.csv": (GEO / "risp_por_bairro.csv", None),
    "aisp_por_bairro.csv": (GEO / "aisp_por_bairro.csv", None),
    "mapeamento_bairros.csv": (GEO / "mapeamento_bairros.csv", None),
    "dicionario_painel_features.csv": (BRUTOS / "dicionario_painel_features.csv", None),
}
ANO0, ANO1 = 2018, 2024
MESES = pd.period_range("%d-01" % ANO0, "%d-12" % ANO1, freq="M")
LAGS = 12
DIAS = {0: "segunda", 1: "terca", 2: "quarta", 3: "quinta", 4: "sexta", 5: "sabado", 6: "domingo"}
# a planilha grava o latrocínio como "ROUBO COM RESULTADO MORTE", sem o sufixo da SSP
NATUREZA = {"HOMICIDIO DOLOSO": "Homicídio Doloso", "FEMINICIDIO": "Feminicídio",
            "LESAO CORPORAL SEGUIDA DE MORTE": "Lesão Corporal Seguida de Morte",
            "ROUBO COM RESULTADO MORTE": "Latrocínio", "ROUBO COM RESULTADO MORTE - LATROCINIO": "Latrocínio"}
MINUSCULAS = {"de", "da", "do", "das", "dos", "e"}
out, NUM = [], []


def p(t=""):
    print(t, flush=True)
    out.append(str(t))


def num(secao, medida, valor, origem):
    NUM.append({"secao": secao, "medida": medida, "valor": valor, "origem": origem})


def br(x, casas=1):
    return ("%.*f" % (casas, x)).replace(".", ",")


def mil(n):
    return "{:,}".format(int(n)).replace(",", ".")


def sha(caminho):
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def titulo(s):
    ps = [w.capitalize() if i == 0 or w.lower() not in MINUSCULAS else w.lower()
          for i, w in enumerate(str(s).split())]
    return " ".join(ps)


def bairro_key(nome):
    n = CRIME_TYPO_FIX.get(norm(nome), norm(nome))
    m = re.match(r"^PARALELA TRECHO (.+)$", n)
    return m.group(1).strip() if m else n


def sem_acento(s):
    return "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn")


def turno(h):
    return "madrugada" if h <= 5 else ("manha" if h <= 11 else ("tarde" if h <= 17 else "noite"))


# ── entradas ─────────────────────────────────────────────────────────────────
p("== entradas ==")
for nome, (cam, esperado) in ENTRADAS.items():
    h = sha(cam)
    p("  %-32s %s" % (nome, h))

cab, corpo, _ = ler_tabela(ENTRADAS["CVLI.xlsx"][0])
b = pd.DataFrame(corpo, columns=cab)
assert len(b) == 7766 and (b.QTD == 1).all()
b["DATA"] = pd.to_datetime(b.DATA)
assert (b.DATA.dt.year == b.ANO).all() and (b.DATA.dt.month == b.MES).all()
assert b.ANO.between(ANO0, ANO1).all()
# ordem cronológica; na mesma data, a ordem em que o registro aparece no CVLI.xlsx
b = b.assign(_ordem_fonte=np.arange(len(b))).sort_values(["DATA", "_ordem_fonte"], kind="stable").reset_index(drop=True)
h = b.HORA
assert h.map(lambda v: isinstance(v, (int, np.integer))).all() and h.between(0, 23).all()
b["NAT"] = b.DELITO.map(lambda v: NATUREZA.get(normalize_delito(v)))
assert b.NAT.notna().all(), b.loc[b.NAT.isna(), "DELITO"].unique()
b["BK"] = b.BAIRRO_C.map(bairro_key)
sexo = b.SEXO.astype(str).str.strip().str.upper()
b["SX"] = np.where(sexo.str.startswith("M"), "M", np.where(sexo.str.startswith("F"), "F", None))
idade = pd.to_numeric(b.IDADE, errors="coerce")
zero_2018 = (b.ANO == 2018) & (idade == 0)
# em 2018 a fonte grava idade ausente como 0 (0 nulos e 51 zeros naquele ano)
b["ID"] = idade.where(~zero_2018)
b["DS"] = b.DATA.dt.dayofweek
p()
p("== CVLI.xlsx ==")
p("  registros: %d | por ano: %s" % (len(b), b.ANO.value_counts().sort_index().to_dict()))
p("  hora inteira de 0 a 23 em todos os registros; sexo não informado: %d; idade 0 de 2018 como ausente: %d"
  % (int(b.SX.isna().sum()), int(zero_2018.sum())))

# ── rótulo de bairro ─────────────────────────────────────────────────────────
mp = pd.read_csv(ENTRADAS["mapeamento_bairros.csv"][0])
malha_norm = {}
for chave, valor in zip(mp.bairro_principal, mp.bairro_socio):
    malha_norm.setdefault(bairro_key(chave), valor)
chaves = sorted(b.BK.unique())
fonte = b.groupby("BK").BAIRRO_C.agg(lambda s: s.mode().iat[0])
rotulo = {k: (malha_norm[k] if k in malha_norm and pd.notna(malha_norm[k]) else titulo(fonte[k])) for k in chaves}
n_malha = sum(1 for k in chaves if k in malha_norm and pd.notna(malha_norm[k]))
contrabarra = [k for k, v in rotulo.items() if "\\" in v]
# contrabarra vira barra normal: parte dos leitores de CSV a toma como escape
rotulo = {k: v.replace("\\", "/") for k, v in rotulo.items()}
assert len(set(rotulo.values())) == len(rotulo) == 166
p()
p("== bairros ==")
p("  bairros: %d | grafia da malha: %d | da fonte em caixa de título: %d | contrabarra trocada: %s"
  % (len(chaves), n_malha, len(chaves) - n_malha, [rotulo[k] for k in contrabarra]))

# ── RISP pela tabela risp_por_bairro.csv ─────────────────────────────────────────────────
pop = pd.read_csv(ENTRADAS["risp_por_bairro.csv"][0], dtype=str, keep_default_na=False)
pop = pop.rename(columns={"bairro": "BAIRRO_P", "risp": "RISP"})
pop["BK"] = pop.BAIRRO_P.map(bairro_key)
assert pop.BK.is_unique and len(pop) == 168
risp_tab = dict(zip(pop.BK, pop.RISP))
sem_risp = [k for k in chaves if k not in risp_tab]
maj = b.groupby("BK")["RISP 2025"].agg(lambda s: s.mode().iat[0])
div_maj = {k: (risp_tab[k], maj[k]) for k in chaves if risp_tab.get(k) != maj[k]}
b["RISP"] = b.BK.map(risp_tab)
div_reg = b[b.RISP != b["RISP 2025"]]
p()
p("== RISP (risp_por_bairro.csv) ==")
p("  bairros na tabela: %d | da base sem RISP na tabela: %s | da tabela fora da base: %s"
  % (len(pop), sem_risp or "nenhum", sorted(set(pop.BK) - set(chaves))))
p("  RISP da tabela contra a maioria por bairro do CVLI.xlsx: %d divergências em %d bairros %s"
  % (len(div_maj), len(chaves), div_maj or ""))
p("  registro a registro contra a coluna RISP 2025: coincide em %d de %d; divergem: %s"
  % (len(b) - len(div_reg), len(b), div_reg.groupby(["BK", "ANO", "RISP 2025"]).size().to_dict()))
assert not sem_risp and not div_maj

# ── AISP pela Portaria 435/2020 ──────────────────────────────────────────────
aisp_tab = mapear_bairros_para_aisp(chaves)
sem_aisp = [k for k in chaves if k not in aisp_tab]
b["AISP"] = b.BK.map(aisp_tab)
risp_por_aisp = pd.DataFrame({"aisp": [aisp_tab[k] for k in chaves], "risp": [risp_tab[k] for k in chaves],
                              "bairro": [rotulo[k] for k in chaves]})
mistas = risp_por_aisp.groupby("aisp").risp.nunique()
mistas = mistas[mistas > 1]
p()
p("== AISP (Portaria SSP 435/2020, Anexo I) ==")
p("  bairros sem AISP: %s" % (sem_aisp or "nenhum"))
p("  AISP distintas: %d | cada AISP numa só RISP: %s" % (len(set(aisp_tab.values())), mistas.empty))
for a in mistas.index:
    g = risp_por_aisp[risp_por_aisp.aisp == a]
    p("    AISP %d dividida: %s" % (a, g.groupby("risp").bairro.apply(list).to_dict()))
p("  RISP por AISP: %s" % risp_por_aisp.groupby("aisp").risp.first().to_dict())
assert not sem_aisp
UMA_RISP = "cada AISP pertence a uma única RISP" if mistas.empty else "%d AISP estão divididas entre RISP" % len(mistas)

# ── a) eventos.csv ───────────────────────────────────────────────────────────
b["TURNO"] = b.HORA.astype(int).map(turno)
e = pd.DataFrame({
    "id_ocorrencia": np.arange(1, len(b) + 1),
    "data": b.DATA.dt.strftime("%Y-%m-%d"),
    "ano": b.ANO.astype(int),
    "mes": b.MES.astype(int),
    "dia_semana": b.DS.map(DIAS),
    "turno": b.TURNO,
    "bairro": b.BK.map(rotulo),
    "risp": b.RISP,
    "aisp": b.AISP.astype(int),
    "tipo_crime": b.NAT,
    "classificacao": "CVLI",
    "sexo_vitima": b.SX,
    "idade_vitima": b.ID.astype("Int64"),
})
e.to_csv(AQUI / "eventos.csv", index=False, encoding="utf-8")
p()
p("== eventos.csv ==")
p("  linhas %d | colunas %d: %s" % (len(e), e.shape[1], list(e.columns)))

# ── c) base bairro-mês ───────────────────────────────────────────────────────
ib = {k: i for i, k in enumerate(chaves)}
bi = b.BK.map(ib).to_numpy()
mi = ((b.ANO - ANO0) * 12 + b.MES - 1).to_numpy()


def grade(peso):
    g = np.zeros((len(chaves), len(MESES)), dtype=np.int64)
    np.add.at(g, (bi, mi), np.asarray(peso, dtype=np.int64))
    return g


X = grade(np.ones(len(b)))
assert X.sum() == len(b) == 7766
CEL = {"n": X, "madrugada": grade(b.TURNO == "madrugada"), "tarde": grade(b.TURNO == "tarde"),
       "noite": grade(b.TURNO == "noite"), "fds": grade(b.DS >= 5), "masc": grade(b.SX == "M"),
       "n_idade": grade(b.ID.notna())}
soma_idade = np.zeros((len(chaves), len(MESES)))
np.add.at(soma_idade, (bi, mi), b.ID.fillna(0).to_numpy(float))
# somas acumuladas inteiras: a janela sai exata, sem o erro de arredondamento da soma móvel em float
ACUM = {c: np.concatenate([np.zeros((len(chaves), 1), dtype=np.int64), np.cumsum(g, axis=1)], axis=1) for c, g in CEL.items()}
ACUM_ID = np.concatenate([np.zeros((len(chaves), 1)), np.cumsum(soma_idade, axis=1)], axis=1)
risp_b = np.array([risp_tab[k] for k in chaves])
aisp_b = np.array([aisp_tab[k] for k in chaves])
rot_b = np.array([rotulo[k] for k in chaves])

linhas, NI, SI = [], [], []
for j in range(LAGS - 1, len(MESES) - 1):
    t, alvo = MESES[j], MESES[j + 1]
    w = {c: ACUM[c][:, j + 1] - ACUM[c][:, j - (LAGS - 1)] for c in ACUM}
    s_id = ACUM_ID[:, j + 1] - ACUM_ID[:, j - (LAGS - 1)]
    n, ni = w["n"], w["n_idade"]
    r = {"bairro": rot_b, "risp": risp_b, "aisp": aisp_b,
         "ano_ref": t.year, "mes_ref": t.month, "ano_alvo": alvo.year, "mes_alvo": alvo.month,
         "month_sin_alvo": np.sin(2 * np.pi * alvo.month / 12), "month_cos_alvo": np.cos(2 * np.pi * alvo.month / 12),
         "year_norm_alvo": (alvo.year - ANO0) / (ANO1 - ANO0), "time_index": j}
    for k in range(1, LAGS + 1):
        r["cvli_lag%d" % k] = X[:, j - k + 1]
    for wj in (3, 6, 12):
        r["cvli_ma%d" % wj] = X[:, j - wj + 1:j + 1].mean(axis=1)
    r["cvli_trend"] = X[:, j] - X[:, j - 1]
    r["n_vitimas_hist12"] = n
    for c in ("madrugada", "tarde", "noite", "fds", "masc"):
        r["prop_%s_hist12" % c] = np.where(n > 0, w[c] / np.where(n > 0, n, 1), np.nan)
    r["idade_media_hist12"] = np.where(ni > 0, s_id / np.where(ni > 0, ni, 1), np.nan)
    r["target_cvli"] = X[:, j + 1]
    linhas.append(pd.DataFrame(r))
    NI.append(ni)
    SI.append(s_id)
NI, SI = np.concatenate(NI), np.concatenate(SI)
F = pd.concat(linhas, ignore_index=True)
ORDEM = (["bairro", "risp", "aisp", "ano_ref", "mes_ref", "ano_alvo", "mes_alvo", "month_sin_alvo", "month_cos_alvo",
          "year_norm_alvo", "time_index"] + ["cvli_lag%d" % k for k in range(1, 13)]
         + ["cvli_ma3", "cvli_ma6", "cvli_ma12", "cvli_trend", "n_vitimas_hist12", "prop_madrugada_hist12",
            "prop_tarde_hist12", "prop_noite_hist12", "prop_fds_hist12", "prop_masc_hist12", "idade_media_hist12",
            "target_cvli"])
F = F[ORDEM]
CRIM = ["cvli_lag%d" % k for k in range(1, 13)] + ["cvli_ma3", "cvli_ma6", "cvli_ma12", "cvli_trend"]
TEMP = ["month_sin_alvo", "month_cos_alvo", "year_norm_alvo", "time_index"]
ESPA = ["bairro", "aisp", "risp"]
PERF = ["n_vitimas_hist12", "prop_madrugada_hist12", "prop_tarde_hist12", "prop_noite_hist12", "prop_fds_hist12",
        "prop_masc_hist12", "idade_media_hist12"]
assert len(CRIM + TEMP + ESPA + PERF) == 30 and set(CRIM + TEMP + ESPA + PERF) <= set(F.columns)
# repr do float: a representação mais curta que volta ao mesmo double
F.to_csv(AQUI / "painel_bairro_mes_features.csv", index=False, encoding="utf-8",
         float_format=lambda v: float(v).__repr__())
part = np.select([F.ano_alvo <= 2022, F.ano_alvo == 2023], ["treino", "validacao"], "teste")
cont_part = {k: int((part == k).sum()) for k in ("treino", "validacao", "teste")}
p()
p("== painel_bairro_mes_features.csv (base bairro-mês) ==")
p("  grade: %d bairros x %d meses = %d células, soma %d" % (len(chaves), len(MESES), X.size, int(X.sum())))
p("  linhas %d (%d meses de alvo x %d bairros) | colunas %d | atributos do modelo 30"
  % (len(F), len(MESES) - LAGS, len(chaves), F.shape[1]))
p("  partição pelo ano do alvo: %s" % cont_part)
assert len(F) == 11952 and cont_part == {"treino": 7968, "validacao": 1992, "teste": 1992}
assert F.time_index.between(0, 83).all()

# ── conferências, sobre os arquivos gravados ─────────────────────────────────
E = pd.read_csv(AQUI / "eventos.csv")
G = pd.read_csv(AQUI / "painel_bairro_mes_features.csv", float_precision="round_trip")
p()
p("== conferências (sobre os arquivos gravados) ==")
G["m"] = (G.ano_ref - ANO0) * 12 + G.mes_ref - 1
nxt = G[["bairro", "m", "cvli_lag1"]].assign(m=G.m - 1)
c1 = G.merge(nxt, on=["bairro", "m"], suffixes=("", "_seg"))
ok1 = bool((c1.target_cvli == c1.cvli_lag1_seg).all())
p("  1. alvo(b,t) = cvli_lag1(b,t+1) em %d pares: %s" % (len(c1), ok1))
E["m"] = (E.ano - ANO0) * 12 + E.mes - 1
cont = E.groupby(["bairro", "m"]).size()
rec = pd.Series(list(zip(G.bairro, G.m + 1))).map(cont).fillna(0).to_numpy()
ok2 = bool((G.target_cvli.to_numpy() == rec).all())
p("  2. alvo = recontagem dos eventos de t+1 em %d linhas: %s" % (len(G), ok2))
por_b = {k: g for k, g in E.groupby("bairro")}
maxd = 0.0
for row in G[["bairro", "m", "n_vitimas_hist12", "prop_madrugada_hist12", "prop_tarde_hist12", "prop_noite_hist12",
              "prop_fds_hist12", "prop_masc_hist12", "idade_media_hist12"]].itertuples(index=False):
    g = por_b.get(row.bairro)
    jan = g[(g.m >= row.m - 11) & (g.m <= row.m)] if g is not None else E.iloc[0:0]
    n_ = len(jan)
    assert n_ == row.n_vitimas_hist12
    calc = [np.nan] * 5 if n_ == 0 else [(jan.turno == "madrugada").mean(), (jan.turno == "tarde").mean(),
                                          (jan.turno == "noite").mean(), jan.dia_semana.isin(["sabado", "domingo"]).mean(),
                                          (jan.sexo_vitima == "M").mean()]
    ida = jan.idade_vitima.dropna()
    calc.append(ida.mean() if len(ida) else np.nan)
    got = [row.prop_madrugada_hist12, row.prop_tarde_hist12, row.prop_noite_hist12, row.prop_fds_hist12,
           row.prop_masc_hist12, row.idade_media_hist12]
    for x, y in zip(calc, got):
        assert np.isnan(x) == np.isnan(y)
        if not np.isnan(x):
            maxd = max(maxd, abs(x - y))
ok3 = maxd < 1e-9
p("  3. proporções e idade média recalculadas só com os eventos da janela t-11..t: maior diferença %.1e: %s" % (maxd, ok3))
dn = float(np.abs(G.n_vitimas_hist12 - 12 * G.cvli_ma12).max())
p("  4. n_vitimas_hist12 = 12 x cvli_ma12: maior desvio %.1e (leitura em precisão plena)" % dn)
p("  5. soma da grade = %d; soma do alvo + defasagens do primeiro mês de referência = %d"
  % (int(X.sum()), int(G.target_cvli.sum() + G[G.m == 11][["cvli_lag%d" % k for k in range(1, 13)]].to_numpy().sum())))
assert ok1 and ok2 and ok3 and X.sum() == 7766

# ── números da base, impressos na tela ────────────────────────────────────────────────
ORIG_E, ORIG_G = "eventos.csv", "painel_bairro_mes_features.csv"
for a, v in e.groupby("ano").size().items():
    num("registros", "registros em %d" % a, mil(v), ORIG_E)
num("registros", "registros, total", mil(len(e)), ORIG_E)
for nat, v in e.tipo_crime.value_counts().items():
    num("naturezas", nat, mil(v), ORIG_E)
ct = pd.crosstab(e.tipo_crime, e.ano)
for nat in ct.index:
    for a in ct.columns:
        num("naturezas por ano", "%s, %d" % (nat, a), mil(ct.loc[nat, a]), ORIG_E)
iv = e.groupby("ano").idade_vitima.apply(lambda s: int(s.isna().sum()))
for a, v in iv.items():
    num("idade", "idade vazia em %d" % a, mil(v), ORIG_E + " (em 2018, os zeros da fonte)")
num("idade", "idade vazia, total", "%s de %s (%s%%)" % (mil(e.idade_vitima.isna().sum()), mil(len(e)),
                                                       br(100 * e.idade_vitima.isna().mean())), ORIG_E)
tur = e.turno.value_counts(normalize=True)
for t_ in ("madrugada", "manha", "tarde", "noite"):
    num("turno", "proporção geral, %s" % t_, "%s%% (%s)" % (br(100 * tur[t_]), mil((e.turno == t_).sum())), ORIG_E)
num("território", "AISP / RISP", "%d / %d" % (len(set(aisp_b)), len(set(risp_b))), ORIG_G)
num("território", "bairros por RISP", ", ".join("%s %d" % (k, v) for k, v in pd.Series(risp_b).value_counts().items()), ORIG_G)
num("base bairro-mês", "células da grade", mil(X.size), "grade 166 x 84")
num("base bairro-mês", "linhas modeláveis", mil(len(G)), ORIG_G)
for k, v in cont_part.items():
    num("base bairro-mês", "linhas de %s" % k, mil(v), ORIG_G + ", ano_alvo")
T, TE = G[part == "treino"], G[part == "teste"]
num("alvo", "zeros no alvo, base modelável", "%s%%" % br(100 * (G.target_cvli == 0).mean()), ORIG_G)
num("alvo", "zeros no alvo, teste 2024", "%s%%" % br(100 * (TE.target_cvli == 0).mean()), ORIG_G)
num("alvo", "média do alvo no teste", br(TE.target_cvli.mean(), 3), ORIG_G)
num("alvo", "máximo do alvo no teste", int(TE.target_cvli.max()), ORIG_G)
num("alvo", "índice de dispersão no treino (variância amostral / média)",
    br(T.target_cvli.var() / T.target_cvli.mean(), 3), ORIG_G)
n0 = G.n_vitimas_hist12 == 0
num("janela", "linhas com n = 0", "%s de %s (%s%%)" % (mil(n0.sum()), mil(len(G)), br(100 * n0.mean())), ORIG_G)
for k in ("treino", "validacao", "teste"):
    s_ = G[part == k].n_vitimas_hist12 == 0
    num("janela", "linhas com n = 0, %s" % k, "%s (%s%%)" % (mil(s_.sum()), br(100 * s_.mean())), ORIG_G)
num("janela", "treino: janelas com até 2 ocorrências (n <= 2, inclui n = 0)", "%s%%" % br(100 * (T.n_vitimas_hist12 <= 2).mean()), ORIG_G)
num("janela", "treino: janelas com 1 ou 2 ocorrências", "%s%%" % br(100 * T.n_vitimas_hist12.between(1, 2).mean()), ORIG_G)
Z = T[CRIM].to_numpy(float)
Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
posto = int(np.linalg.matrix_rank(Z))
L = Z[:, :12]
vifs = []
for jj in range(12):
    A = np.column_stack([np.ones(len(L)), np.delete(L, jj, axis=1)])
    coef, *_ = np.linalg.lstsq(A, L[:, jj], rcond=None)
    r2 = 1 - np.sum((L[:, jj] - A @ coef) ** 2) / np.sum(L[:, jj] ** 2)
    vifs.append(1 / max(1 - r2, 1e-12))
num("bloco criminal", "posto dos 16 atributos criminais (treino)", "%d de 16" % posto, ORIG_G)
num("bloco criminal", "VIF", "infinito nos 16 (cvli_ma3, cvli_ma6, cvli_ma12 e cvli_trend são combinação exata das defasagens)", ORIG_G)
num("bloco criminal", "VIF máximo entre as 12 defasagens (treino)", br(max(vifs), 2), ORIG_G)
num("bloco criminal", "correlação máxima entre pares de defasagens (treino)", br(np.abs(np.corrcoef(L.T) - np.eye(12)).max(), 3), ORIG_G)
num("bloco criminal", "maior desvio entre n_vitimas_hist12 e 12 x cvli_ma12", "%.1e" % dn, ORIG_G)
ok = T.prop_masc_hist12.notna()
num("perfil", "Spearman entre prop_masc_hist12 e n_vitimas_hist12 (treino, n > 0)",
    br(spearmanr(T.loc[ok, "prop_masc_hist12"], T.loc[ok, "n_vitimas_hist12"]).statistic, 3), ORIG_G)
# a fórmula do dicionário de origem divide a soma das idades pelo total de vítimas da janela
nv = F.n_vitimas_hist12.to_numpy()
com_ambas = NI > 0
nossa = np.where(com_ambas, SI / np.where(com_ambas, NI, 1), np.nan)
de_origem = np.where(nv > 0, SI / np.where(nv > 0, nv, 1), np.nan)
muda = com_ambas & (np.abs(nossa - de_origem) > 1e-9)
pct_muda = 100 * muda.sum() / com_ambas.sum()
num("perfil", "idade média: linhas em que a fórmula do dicionário de origem difere",
    "%s%% (%s de %s com as duas fórmulas definidas)" % (br(pct_muda), mil(muda.sum()), mil(com_ambas.sum())), ORIG_G)
p()
p("== números da base ==")
for r_ in NUM:
    p("  [%s] %s: %s" % (r_["secao"], r_["medida"], r_["valor"]))

# ── b) e d) dicionários ──────────────────────────────────────────────────────
N = len(e)
n_id_vazia = int(e.idade_vitima.isna().sum())
DP = pd.read_csv(ENTRADAS["dicionario_painel_features.csv"][0])
n_vazia = int(n0.sum())
aj_p = {    # coluna: (descrição, fórmula); None mantém o texto do dicionário de origem
    "bairro": ("Bairro ao qual a observação se refere. %d bairros, com a mesma grafia de eventos.csv. Única alteração de grafia em relação à malha do IBGE: em 'Vila Ruy Barbosa/Jardim Cruzeiro' a contrabarra foi trocada por barra normal, porque parte dos leitores de CSV interpreta a contrabarra como escape." % len(chaves), None),
    "risp": ("RISP do bairro (BTS, CENTRAL ou ATLÂNTICO), pela tabela bairro → RISP do POPULAÇAO.xlsx. Constante por bairro.", None),
    "year_norm_alvo": (None, "(ano_alvo - ano_min) / (ano_max - ano_min), com ano_min = 2018 e ano_max = 2024, o primeiro e o último ano da série de eventos."),
    "time_index": ("Índice sequencial do mês de referência t na série.", "0 em 2018-01, 83 em 2024-12."),
    "n_vitimas_hist12": ("Número de CVLI do bairro nos 12 meses anteriores ao alvo; é o denominador das proporções de perfil. Nesta base bairro-mês equivale a 12 x cvli_ma12 por construção (mesma janela); lido em precisão plena, %s." % ("o desvio é zero" if dn == 0 else "o maior desvio é %.1e" % dn), None),
    "prop_fds_hist12": ("Proporção de CVLI em fim de semana na janela de 12 meses. Fim de semana = sábado e domingo. Vazia quando n_vitimas_hist12 = 0.", None),
    "prop_masc_hist12": ("Proporção de vítimas masculinas na janela de 12 meses. Vazia quando n_vitimas_hist12 = 0.", None),
    "idade_media_hist12": ("Idade média das vítimas na janela de 12 meses. Vazia quando nenhuma vítima da janela tem idade informada.",
                           "(Soma das idades informadas nos últimos 12 meses) / (número de vítimas com idade informada). Difere do dicionário de origem, que divide pelo total de vítimas; a diferença aparece em %s%% das linhas em que as duas fórmulas têm valor (%s de %s), porque %s%% dos registros não têm idade. Idade 0 de 2018 tratada como ausente." % (br(pct_muda), mil(muda.sum()), mil(com_ambas.sum()), br(100 * n_id_vazia / N))),
}
TURNOS = [("madrugada", "00:00 a 05:59"), ("tarde", "12:00 a 17:59"), ("noite", "18:00 a 23:59")]
linhas_p = []
for _, r_ in DP.iterrows():
    col, grupo, desc, form, disp = r_.coluna, r_.grupo, r_.descricao, r_.formula_ou_calculo, r_.disponivel_no_momento_da_previsao
    if col == "prop_noturno_hist12":
        for nome_t, faixa in TURNOS:
            linhas_p.append({"coluna": "prop_%s_hist12" % nome_t, "grupo": grupo,
                             "descricao": "Proporção de CVLI no turno da %s (%s) na janela de 12 meses, com a hora inteira, de 0 a 23, tomada como hora cheia. A manhã (06:00 a 11:59) fica fora, como referência. Substitui prop_noturno_hist12 do dicionário de origem. Vazia quando n_vitimas_hist12 = 0 (%s de %s linhas)." % (nome_t, faixa, mil(n_vazia), mil(len(G))),
                             "formula_ou_calculo": "(Σ CVLI no turno da %s nos últimos 12 meses) / (Σ CVLI nos últimos 12 meses)" % nome_t,
                             "disponivel_no_momento_da_previsao": disp})
        continue
    if col in aj_p:
        d_, f_ = aj_p[col]
        desc = d_ or desc
        form = f_ if f_ is not None else form
    linhas_p.append({"coluna": col, "grupo": grupo, "descricao": desc, "formula_ou_calculo": form,
                     "disponivel_no_momento_da_previsao": disp})
    if col == "risp":
        linhas_p.append({"coluna": "aisp", "grupo": "Espacial",
                         "descricao": "AISP do bairro (%d AISP, Portaria SSP 435/2020, Anexo I). Coluna acrescentada ao dicionário de origem: bairro, aisp e risp entram juntos no bloco espacial. Constante por bairro; %s." % (len(set(aisp_b)), UMA_RISP),
                         "formula_ou_calculo": np.nan, "disponivel_no_momento_da_previsao": "Sim"})
DP2 = pd.DataFrame(linhas_p)
assert list(DP2.coluna) == list(F.columns), (list(DP2.coluna), list(F.columns))
DP2.to_csv(AQUI / "dicionario_painel_features.csv", index=False, encoding="utf-8")
txt = (AQUI / "dicionario_painel_features.csv").read_text(encoding="utf-8")
assert "0h-" not in txt and "painel congelado" not in txt
p()
p("== dicionários ==")
p("  dicionario_painel_features.csv: %d linhas, na ordem das colunas do arquivo" % len(DP2))
