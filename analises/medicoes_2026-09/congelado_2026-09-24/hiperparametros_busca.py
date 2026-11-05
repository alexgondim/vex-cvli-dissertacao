"""Busca de hiperparametros na validacao de 2023 (despacho de 24/09, item 11 da proposta).

Protocolo gravado antes em docs/auditoria/hiperparametros_2026-09-24.md.
Doze dobras mensais em 2023: cada uma treina com todos os alvos anteriores ao mes e
preve o mes. O painel e cortado em 2023-12 antes de qualquer ajuste. Criterio:
deviance de Poisson (piso 0,01) somada nas 1.992 linhas de validacao; empate abaixo
de 0,0001 vai para a configuracao mais simples. Sem parada antecipada, semente 42,
threads no padrao das bibliotecas.

Cada configuracao e gravada ao terminar (busca_<estimador>.csv); uma configuracao ja
gravada e pulada, entao a busca pode ser retomada.

Uso: python hiperparametros_busca.py lgbm_poisson | lgbm_tweedie | xgb
"""
import itertools
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")
AQUI = Path(__file__).resolve().parent
CONG = AQUI / "congelado_2026-09-22"
SAIDA = AQUI / "congelado_2026-09-24"
SAIDA.mkdir(exist_ok=True)
LAGS = 12
AUTO = ["cvli_lag%d" % k for k in range(1, LAGS + 1)] + ["ma3", "ma6", "ma12", "trend"]
TEMPO = ["month_sin_alvo", "month_cos_alvo", "year_norm_alvo", "time_index"]
ESPACO = ["bairro", "AISP"]
PERFIL = ["prop_noturno", "prop_fds", "prop_masc", "idade_media", "n_vitimas_hist12"]
FEATS = AUTO + TEMPO + ESPACO + PERFIL
CLIP = 0.01
EST = sys.argv[1]

LGB = list(itertools.product([0.05, 0.1], [100, 300], [7, 15, 31], [20, 50, 100]))
GRADE = {
    "lgbm_poisson": [dict(learning_rate=a, n_estimators=b, num_leaves=c, min_child_samples=d) for a, b, c, d in LGB],
    "lgbm_tweedie": [dict(learning_rate=a, n_estimators=b, num_leaves=c, min_child_samples=d, tweedie_variance_power=e)
                     for (a, b, c, d), e in itertools.product(LGB, [1.2, 1.5, 1.8])],
    "xgb": [dict(learning_rate=a, n_estimators=b, max_depth=c, min_child_weight=d)
            for a, b, c, d in itertools.product([0.05, 0.1, 0.3], [100, 300], [3, 6], [1, 5, 20])],
}
PADRAO = {"lgbm_poisson": dict(learning_rate=0.1, n_estimators=100, num_leaves=31, min_child_samples=20),
          "lgbm_tweedie": dict(learning_rate=0.1, n_estimators=100, num_leaves=31, min_child_samples=20,
                               tweedie_variance_power=1.5),
          "xgb": dict(learning_rate=0.3, n_estimators=100, max_depth=6, min_child_weight=1)}
assert PADRAO[EST] in GRADE[EST]


def modelo(est, h):
    if est == "lgbm_poisson":
        return LGBMRegressor(objective="poisson", random_state=42, verbose=-1, **h)
    if est == "lgbm_tweedie":
        return LGBMRegressor(objective="tweedie", random_state=42, verbose=-1, **h)
    return xgb.XGBRegressor(objective="count:poisson", random_state=42, enable_categorical=True, **h)


def dev_soma(y_, q):
    q = np.clip(q, CLIP, None)
    t = np.where(y_ > 0, y_ * np.log(np.where(y_ > 0, y_, 1) / q), 0.0)
    return float(2 * np.sum(t - (y_ - q)))


P = pd.read_csv(CONG / "painel_mensal.csv")              # leitura igual a do script congelado
P["PER_ALVO"] = pd.PeriodIndex(P.PER_ALVO, freq="M")
P = P[P.PER_ALVO <= pd.Period("2023-12", "M")].copy()    # nenhuma linha de 2024 ou 2025 segue adiante
assert P.PER_ALVO.max() == pd.Period("2023-12", "M")
for c in ESPACO:
    P[c] = P[c].astype("category")
dobras = [pd.Period("2023-%02d" % i, "M") for i in range(1, 13)]
DADOS = [(P[P.PER_ALVO < m], P[P.PER_ALVO == m]) for m in dobras]
assert sum(len(te) for _, te in DADOS) == 1992

arq = SAIDA / ("busca_%s.csv" % EST)
feitas = set()
if arq.exists():
    feitas = set(pd.read_csv(arq).config_json)
for i, h in enumerate(GRADE[EST]):
    chave = json.dumps(h, sort_keys=True)
    if chave in feitas:
        continue
    t0 = time.perf_counter()
    ys, qs = [], []
    for tr, te in DADOS:
        mod = modelo(EST, h).fit(tr[FEATS], tr.alvo.to_numpy(float))
        qs.append(mod.predict(te[FEATS]))
        ys.append(te.alvo.to_numpy(float))
    y, q = np.concatenate(ys), np.concatenate(qs)
    reg = dict(estimador=EST, ordem_grade=i, config_json=chave, **h, padrao=(h == PADRAO[EST]),
               deviance_soma=dev_soma(y, q), deviance_media=dev_soma(y, q) / len(y),
               mae=float(np.mean(np.abs(y - q))), rmse=float(np.sqrt(np.mean((y - q) ** 2))),
               pred_media=float(q.mean()), obs_media=float(y.mean()), tempo_s=time.perf_counter() - t0)
    pd.DataFrame([reg]).to_csv(arq, mode="a", header=not arq.exists(), index=False)
    print("%s %3d/%d  deviance somada %.4f  %.1f s  %s" % (EST, i + 1, len(GRADE[EST]), reg["deviance_soma"],
                                                         reg["tempo_s"], chave), flush=True)

# escolha, pelo criterio do protocolo
R = pd.read_csv(arq)
assert len(R) == len(GRADE[EST])
melhor = R.deviance_soma.min()
cand = R[R.deviance_soma - melhor < 0.0001].copy()
simples = ["num_leaves", "n_estimators"] if EST != "xgb" else ["max_depth", "n_estimators"]
esc = cand.sort_values(simples + ["ordem_grade"]).iloc[0]
R = R.sort_values(["deviance_soma", "ordem_grade"]).reset_index(drop=True)
pos_padrao = int(R.index[R.padrao].tolist()[0]) + 1
h = json.loads(esc.config_json)
(SAIDA / ("escolha_%s.json" % EST)).write_text(json.dumps(dict(
    estimador=EST, escolhida=h, coincide_com_padrao=(h == PADRAO[EST]), deviance_soma=float(esc.deviance_soma),
    empatadas_no_limiar=int(len(cand)), posicao_do_padrao=pos_padrao, configuracoes=int(len(R)),
    deviance_soma_padrao=float(R[R.padrao].deviance_soma.iloc[0])), indent=1), encoding="utf-8")
print("ESCOLHIDA %s: %s | deviance somada %.4f | empatadas no limiar %d | padrao na posicao %d de %d"
      % (EST, h, esc.deviance_soma, len(cand), pos_padrao, len(R)), flush=True)
