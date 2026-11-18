"""Partes comuns dos experimentos: modo de execução (ensaio ou teste), leitura da base, as quatro
referências de comparação (persistência, média móvel de 12 meses, média sazonal e binomial negativa NB2), métricas e
gravação.

Importa `modelo/comum.py` (base, atributos, estimador, métricas) e lê `resultados/escolha.json` (a configuração do
modelo). Os programas que o importam gravam em `resultados/`, no modo `teste`, e em `resultados/ensaio/`, no modo
`ensaio`.

No modo `ensaio`, as linhas de alvo em 2024 saem da base logo depois da leitura, e o ano avaliado é 2023; no modo
`teste`, o ano avaliado é 2024. Os programas rodam da raiz do repositório, por exemplo:
python experimentos/validacao_corte_fixo.py teste
"""
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import digamma, gammaln

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "modelo"))
import comum  # noqa: E402

# as saídas de modelo/ e de experimentos/ ficam todas em resultados/
RESULTADOS = comum.PASTA
PASTA_ENSAIO = RESULTADOS / "ensaio"
CONFIG = json.loads((RESULTADOS / "escolha.json").read_text(encoding="utf-8"))["configuracao"]
ESTIMATIVAS = RESULTADOS / "contribuicoes_corte_fixo.csv"
ALVO = comum.ALVO
METODOS = ("modelo", "binomial_negativa", "persistencia", "media_movel_12", "media_sazonal")
ANO = {"ensaio": 2023, "teste": 2024}
JANELA = {"ensaio": pd.period_range("2022-01", "2023-12", freq="M"),
          "teste": pd.period_range("2022-01", "2024-12", freq="M")}
ANOS_DE_TREINO = (2019, 2022)  # corte fixo: alvos de 2019 a 2022
# binomial negativa: as 12 defasagens e os 4 temporais, padronizados, e a AISP em indicadoras, sem a primeira
NB_NUMERICAS = ["cvli_lag%d" % k for k in range(1, 13)] + comum.BLOCOS["temporal"]
NB_MAXITER = 10000
NB_GTOL = 1e-10
NB_TOL_GRADIENTE = 1e-6  # maior |gradiente| da log-verossimilhança média, no fim, para contar como convergência
NB_RECOMECOS = 2


def agora():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def registrar(texto):
    linha = "%s  %s" % (agora(), texto)
    print(linha, flush=True)


def sha256(caminho):
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def modo():
    """ensaio ou teste, da linha de comando."""
    assert len(sys.argv) == 2 and sys.argv[1] in ANO, "uso: programa.py ensaio|teste"
    m = sys.argv[1]
    return m


def pasta_de_saida(m):
    pasta = RESULTADOS if m == "teste" else PASTA_ENSAIO
    pasta.mkdir(exist_ok=True)
    return pasta


def gravar_csv(tabela, caminho):
    """Grava, sobrescrevendo o que houver."""
    with open(caminho, "w", encoding="utf-8", newline="") as f:
        tabela.to_csv(f, index=False)


def gravar_json(dado, caminho):
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dado, f, ensure_ascii=False, indent=1)


def ler_base(m):
    """Base por bairro e mês. No ensaio, sem as linhas de alvo em 2024."""
    b = comum.ler_base()
    if m == "ensaio":
        b = b[b["ano_alvo"] <= 2023].copy()
        assert (b["ano_alvo"] <= 2023).all()
    return b


def media_sazonal(base):
    """Para cada linha, a média das contagens do bairro no mesmo mês do alvo, em todos os anos anteriores ao do alvo,
    desde 2018. A contagem de (bairro, ano, mês) é o alvo da linha desse mês; a de 2018, que não é alvo de linha
    nenhuma, é a cvli_lag12 da linha do mesmo mês de 2019."""
    contagem = base[["bairro", "ano_alvo", "mes_alvo", ALVO]].rename(columns={"ano_alvo": "ano", "mes_alvo": "mes",
                                                                            ALVO: "n"})
    de_2018 = base.loc[base["ano_alvo"] == 2019, ["bairro", "mes_alvo", "cvli_lag12"]].rename(
        columns={"mes_alvo": "mes", "cvli_lag12": "n"}).assign(ano=2018)
    contagem = pd.concat([de_2018[["bairro", "ano", "mes", "n"]], contagem], ignore_index=True)
    contagem["bairro"] = contagem["bairro"].astype(str)
    tabela = contagem.pivot_table(index=["bairro", "mes"], columns="ano", values="n", aggfunc="sum")
    saida = np.empty(len(base))
    for i, (b, a, mes) in enumerate(zip(base["bairro"].astype(str), base["ano_alvo"], base["mes_alvo"])):
        anteriores = tabela.loc[(b, mes), [y for y in tabela.columns if y < a]]
        assert anteriores.notna().all() and len(anteriores) == a - 2018
        saida[i] = anteriores.mean()
    return pd.Series(saida, index=base.index)


def referencias(base):
    """As 3 referências ingênuas de cada linha: persistência (contagem do mês de referência, cvli_lag1), média
    móvel de 12 meses (cvli_ma12) e média sazonal."""
    return pd.DataFrame({"persistencia": base["cvli_lag1"].astype(float),
                         "media_movel_12": base["cvli_ma12"].astype(float),
                         "media_sazonal": media_sazonal(base)}, index=base.index)


def desenho_nb(df, padronizacao, aisps):
    """Intercepto, as 16 numéricas padronizadas pela média e pelo desvio do treino do ajuste e as indicadoras da
    AISP sem a primeira categoria."""
    media, desvio = padronizacao
    numericas = (df[NB_NUMERICAS].to_numpy(float) - media) / desvio
    aisp = df["aisp"].astype(str).to_numpy()
    indicadoras = np.column_stack([(aisp == a).astype(float) for a in aisps[1:]])
    return np.column_stack([np.ones(len(df)), numericas, indicadoras])


def nb2_negativa(par, X, y):
    """Menos a log-verossimilhança média da NB2 (variância mu + mu²/r, ligação log) e o gradiente analítico;
    par = (beta, log r)."""
    beta, log_r = par[:-1], par[-1]
    r = np.exp(log_r)
    eta = X @ beta
    mu = np.exp(eta)
    log_r_mu = np.log(r + mu)
    ll = gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * (log_r - log_r_mu) + y * (eta - log_r_mu)
    g_beta = X.T @ ((y - mu) * r / (r + mu))
    g_log_r = (digamma(y + r) - digamma(r) + log_r + 1 - log_r_mu - (y + r) / (r + mu)).sum() * r
    n = len(y)
    return -ll.sum() / n, -np.concatenate([g_beta, [g_log_r]]) / n


def conferir_gradiente(X, y, par):
    """Erro relativo máximo entre o gradiente analítico e o numérico (diferença central, passo 1e-6)."""
    _, g = nb2_negativa(par, X, y)
    numerico = np.array([(nb2_negativa(par + e, X, y)[0] - nb2_negativa(par - e, X, y)[0]) / 2e-6
                         for e in np.eye(len(par)) * 1e-6])
    return float(np.abs(g - numerico).max() / max(np.abs(numerico).max(), 1e-12))


def ajustar_nb(treino):
    """Máxima verossimilhança da NB2 por L-BFGS-B (scipy), partindo de beta = 0, intercepto = log da média do alvo
    e log r = 0. Converge quando o maior |gradiente| da log-verossimilhança média, no ponto final, fica abaixo de
    NB_TOL_GRADIENTE; a bandeira de sucesso do otimizador não conta, porque com ftol = 0 ele costuma parar na busca
    de linha já no ótimo numérico. Senão, recomeça do último ponto até NB_RECOMECOS vezes e, se ainda não convergir,
    fica com o último ponto e registra a não convergência."""
    y = treino[ALVO].to_numpy(float)
    numericas = treino[NB_NUMERICAS].to_numpy(float)
    padronizacao = (numericas.mean(axis=0), numericas.std(axis=0))
    aisps = sorted(treino["aisp"].astype(str).unique(), key=int)
    X = desenho_nb(treino, padronizacao, aisps)
    assert np.linalg.matrix_rank(X) == X.shape[1], "desenho da binomial negativa sem posto completo"
    par = np.zeros(X.shape[1] + 1)
    par[0] = np.log(y.mean())
    iteracoes, recomecos = 0, 0
    while True:
        r = minimize(nb2_negativa, par, args=(X, y), jac=True, method="L-BFGS-B",
                     options={"maxiter": NB_MAXITER, "maxfun": 2 * NB_MAXITER, "gtol": NB_GTOL, "ftol": 0.0})
        par, iteracoes = r.x, iteracoes + r.nit
        gradiente = float(np.abs(r.jac).max())
        convergiu = gradiente < NB_TOL_GRADIENTE
        if convergiu or recomecos == NB_RECOMECOS:
            break
        recomecos += 1
    return {"par": par, "padronizacao": padronizacao, "aisps": aisps, "convergiu": convergiu,
            "iteracoes": int(iteracoes), "recomecos": recomecos, "gradiente_max": gradiente,
            "r": float(np.exp(par[-1])), "motivo": str(r.message), "sucesso_do_otimizador": bool(r.success),
            "linhas_de_treino": int(len(y))}


def prever_nb(ajuste, df):
    X = desenho_nb(df, ajuste["padronizacao"], ajuste["aisps"])
    return np.exp(X @ ajuste["par"][:-1])


def resumo_nb(ajuste, rotulo):
    return {"ajuste": rotulo, "convergiu": ajuste["convergiu"], "iteracoes": ajuste["iteracoes"],
            "recomecos": ajuste["recomecos"], "gradiente_max": ajuste["gradiente_max"], "r": ajuste["r"],
            "alpha": 1 / ajuste["r"], "motivo": ajuste["motivo"], "sucesso_do_otimizador": ajuste["sucesso_do_otimizador"],
            "linhas_de_treino": ajuste["linhas_de_treino"]}


def linha_de_metricas(y, q, **chaves):
    m = comum.metricas(y, q)
    return {**chaves, "linhas": int(len(y)), "mae": m["mae"], "rmse": m["rmse"], "deviance": m["deviance"]}
