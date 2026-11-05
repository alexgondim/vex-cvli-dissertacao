"""Item 10 do bloco 31 (modo ensaio, só 2023): importância por permutação, nas 1.992 linhas de 2023, com as mesmas
5 permutações para tudo.

LSTM (lstm_modelo.pt): embaralha a contagem de cada um dos 12 passos, um de cada vez, mantendo o seno e o cosseno.
M4 das 12 dobras: reajusta os 12 modelos da Parte 1 (têm de repetir bit a bit as previsões gravadas lá), embaralha
cada um dos 30 atributos, um de cada vez, e prevê cada linha com o modelo da dobra dela. Importância = média do
aumento da deviance nas 5 permutações.

Grava permutacao_lstm_2023.csv, permutacao_m4_2023.csv e resumo_lstm.json (ordenações das 12 defasagens e tau-b).
"""
import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau

import comum3 as c

m = c.modo()
assert m == "ensaio"
c.registrar("ensaio, item 10: início; base lida sem as linhas de 2024")
base = c.ler_base(m)
avaliadas = base[base["ano_alvo"] == 2023]
y = avaliadas[c.ALVO].to_numpy(float)
rng = np.random.default_rng(c.SEMENTE_PERMUTACAO)
permutacoes = [rng.permutation(len(avaliadas)) for _ in range(c.PERMUTACOES)]


def aumentos(prever_com, referencia):
    return [c.comum.metricas(y, prever_com(p))["deviance"] - referencia for p in permutacoes]


# LSTM
rede = c.Rede()
rede.load_state_dict(torch.load(c.PARTE3 / "lstm_modelo.pt"))
X, _ = c.sequencias(avaliadas)
dev_lstm = c.comum.metricas(y, c.prever(rede, X))["deviance"]
ensaio = pd.read_csv(c.PARTE3 / "lstm_sementes.csv").set_index("semente").loc[c.SEMENTE_TESTE, "deviance"]
assert dev_lstm == ensaio, "a deviance da LSTM em 2023 não repete a do ensaio"
linhas_lstm = []
for k in range(c.PASSOS):
    def com_passo(p, k=k):
        Xp = X.copy()
        Xp[:, k, 0] = X[p, k, 0]
        return c.prever(rede, Xp)
    a = aumentos(com_passo, dev_lstm)
    linhas_lstm.append({"passo": k + 1, "defasagem": "cvli_lag%d" % (c.PASSOS - k), "aumento_medio": float(np.mean(a)),
                        **{"aumento_%d" % (i + 1): v for i, v in enumerate(a)}})
lstm = pd.DataFrame(linhas_lstm)
lstm["ordem_12"] = lstm["aumento_medio"].rank(ascending=False, method="min").astype(int)
c.c2.gravar_csv(lstm, c.PARTE3 / "permutacao_lstm_2023.csv")
c.registrar("item 10, LSTM: deviance sem permutação %.6f; ordem das defasagens pelo aumento: %s"
            % (dev_lstm, ", ".join(lstm.sort_values("aumento_medio", ascending=False)["defasagem"])))

# M4 das 12 dobras: reajuste e conferência contra a Parte 1
modelos = {}
partes = []
for mes, treino, teste in c.comum.dobras_2023(base):
    modelos[mes] = c.comum.estimador(c.CONFIG).fit(base.loc[treino, c.M4], base.loc[treino, c.ALVO].to_numpy(float))
    partes.append(pd.Series(modelos[mes].predict(base.loc[teste, c.M4]), index=base.index[teste]))
previsao = pd.concat(partes).loc[avaliadas.index]
p1 = pd.read_csv(c.RODADA / "previsoes_ablacao_sensibilidade.csv")
p1 = p1[p1["modelo"] == "M4"]["previsao"].to_numpy().astype(np.float32)
escolha = json.loads(c.c2.RODADA.joinpath("escolha.json").read_text(encoding="utf-8"))
m4 = c.comum.metricas(y, previsao.to_numpy(float))
iguais = bool(np.array_equal(p1, previsao.to_numpy())) and all(m4[k] == escolha[k] for k in ("mae", "rmse", "deviance"))
c.registrar("item 10, M4 das 12 dobras reajustado: previsões iguais bit a bit (float32) às da Parte 1 e métricas "
            "iguais às de escolha.json: %s; deviance %.6f" % (iguais, m4["deviance"]))
assert iguais, "o M4 reajustado não repete a Parte 1"
dobra_de = pd.Series({i: mes for mes, _, teste in c.comum.dobras_2023(base) for i in base.index[teste]}).loc[avaliadas.index]
Xa = avaliadas[c.M4].copy()


def prever_m4(X):
    saida = pd.Series(np.nan, index=X.index)
    for mes, modelo in modelos.items():
        i = dobra_de.index[dobra_de == mes]
        saida.loc[i] = modelo.predict(X.loc[i])
    assert saida.notna().all()
    return saida.to_numpy(float)


assert (prever_m4(Xa) == previsao.to_numpy()).all()
linhas_m4 = []
for atributo in c.M4:
    def com_atributo(p, atributo=atributo):
        Xp = Xa.copy()
        valores = Xa[atributo].to_numpy()[p]
        if atributo in c.comum.CATEGORICAS:
            Xp[atributo] = pd.Categorical(valores, categories=Xa[atributo].cat.categories)
        else:
            Xp[atributo] = valores
        return prever_m4(Xp)
    a = aumentos(com_atributo, m4["deviance"])
    linhas_m4.append({"atributo": atributo, "aumento_medio": float(np.mean(a)),
                      **{"aumento_%d" % (i + 1): v for i, v in enumerate(a)}})
tab = pd.DataFrame(linhas_m4)
tab["ordem_30"] = tab["aumento_medio"].rank(ascending=False, method="min").astype(int)
defasagens = tab["atributo"].isin(c.DEFASAGENS)
tab["ordem_12"] = tab.loc[defasagens, "aumento_medio"].rank(ascending=False, method="min").astype(int).reindex(tab.index)
c.c2.gravar_csv(tab, c.PARTE3 / "permutacao_m4_2023.csv")
c.registrar("item 10, M4: ordem geral dos 30 pelo aumento: %s"
            % ", ".join(tab.sort_values("aumento_medio", ascending=False)["atributo"]))

# tau-b entre as 2 ordenações das 12 defasagens
a_lstm = lstm.set_index("defasagem").loc[c.DEFASAGENS, "aumento_medio"].to_numpy()
a_m4 = tab.set_index("atributo").loc[c.DEFASAGENS, "aumento_medio"].to_numpy()
tau, _ = kendalltau(a_lstm, a_m4, variant="b")
c.registrar("item 10, tau-b entre as ordenações das 12 defasagens (LSTM x M4): %.6f" % tau)
c.c2.gravar_json({"linhas": 1992, "permutacoes": c.PERMUTACOES, "semente_permutacao": c.SEMENTE_PERMUTACAO,
                  "deviance_sem_permutacao": {"lstm": dev_lstm, "m4_dobras": m4["deviance"]},
                  "ordem_12_lstm": lstm.sort_values("aumento_medio", ascending=False)["defasagem"].tolist(),
                  "ordem_12_m4": tab[defasagens].sort_values("aumento_medio", ascending=False)["atributo"].tolist(),
                  "ordem_30_m4": tab.sort_values("aumento_medio", ascending=False)["atributo"].tolist(),
                  "posicao_das_12_defasagens_entre_os_30_m4": {r.atributo: int(r.ordem_30) for r in tab[defasagens].itertuples()},
                  "tau_b_12_defasagens_lstm_x_m4": float(tau)}, c.PARTE3 / "resumo_lstm.json")
c.registrar("ensaio, item 10: fim; gravados permutacao_lstm_2023.csv, permutacao_m4_2023.csv e resumo_lstm.json")
