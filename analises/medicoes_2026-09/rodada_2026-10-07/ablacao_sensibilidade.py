"""Itens 5 e 6 do bloco 26: ablação e sensibilidade territorial, nas 12 dobras de 2023, com a configuração escolhida.

Ablação: M1 criminal (16), M2 + temporal (20), M3 + espacial (23), M4 + perfil (30); o ganho de cada bloco é a
métrica do modelo menos a do anterior (negativo é melhora). O M4 tem de repetir a busca; se não repetir, para.

Sensibilidade: M4 (bairro, aisp e risp), MA (só bairro), MB (só risp) e MC (bairro e risp), sempre com os outros
27 atributos. Para cada um: as 3 métricas, as contribuições das 1.992 linhas (cada linha pelo modelo da dobra
dela) e a média do valor absoluto da contribuição dos 27 atributos não espaciais. Depois, o tau-b de Kendall entre
as 4 ordenações dos 27, nos 6 pares, e a repartição do peso espacial no M4.
"""
import itertools
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import comum

escolha = json.loads(comum.ESCOLHA.read_text(encoding="utf-8"))
config = escolha["configuracao"]
base = comum.ler_base()
B = comum.BLOCOS
NAO_ESPACIAIS = comum.em_ordem(B["criminal"] + B["temporal"] + B["perfil"])
assert len(NAO_ESPACIAIS) == 27
ABLACAO = {"M1": B["criminal"], "M2": B["criminal"] + B["temporal"], "M3": B["criminal"] + B["temporal"] + B["espacial"],
           "M4": comum.atributos()}
SENSIBILIDADE = {"M4": comum.atributos(), "MA": NAO_ESPACIAIS + ["bairro"], "MB": NAO_ESPACIAIS + ["risp"],
                 "MC": NAO_ESPACIAIS + ["bairro", "risp"]}

comum.registrar("etapa 4, ablação: início, configuração %d %s" % (escolha["ordem"], config))
previsoes, contribs, metricas = [], {}, {}
for nome in ["M1", "M2", "M3", "M4", "MA", "MB", "MC"]:
    nomes = ABLACAO.get(nome) or SENSIBILIDADE[nome]
    prev, c = comum.prever_dobras(base, nomes, config, com_contribuicoes=nome in SENSIBILIDADE)
    metricas[nome] = comum.metricas(prev[comum.ALVO], prev["previsao"])
    prev.insert(0, "modelo", nome)
    previsoes.append(prev.rename(columns={comum.ALVO: "alvo"}))
    if c is not None:
        contribs[nome] = c
    comum.registrar("%s (%d atributos): MAE %.6f, RMSE %.6f, deviance %.6f" % (nome, len(comum.em_ordem(nomes)),
                                                                               *metricas[nome].values()))
    if nome == "M4":
        # o XGBoost prevê em float32, e o CSV guarda o texto desse valor: a comparação volta a float32, senão
        # o arredondamento do texto aparece como diferença
        busca = pd.read_csv(comum.PASTA / "previsoes_busca.csv")
        busca = busca[busca["ordem"] == escolha["ordem"]]["previsao"].to_numpy().astype(np.float32)
        previsao_igual = bool(np.array_equal(busca, prev["previsao"].to_numpy()))
        metricas_iguais = all(metricas["M4"][k] == escolha[k] for k in ("mae", "rmse", "deviance"))
        comum.registrar("M4 contra a busca: previsões iguais bit a bit (float32): %s; MAE, RMSE e deviance iguais: %s"
                        % (previsao_igual, metricas_iguais))
        if not (previsao_igual and metricas_iguais):
            comum.registrar("PARADA: o M4 não repete a configuração escolhida na busca")
            sys.exit(1)
        comum.registrar("etapa 5, sensibilidade territorial: início")

pd.concat(previsoes).to_csv(comum.PASTA / "previsoes_ablacao_sensibilidade.csv", index=False)
tab = pd.DataFrame([dict(modelo=n, atributos=len(comum.em_ordem(ABLACAO[n])), **metricas[n]) for n in ABLACAO])
for k in ("mae", "rmse", "deviance"):
    tab["ganho_" + k] = tab[k].diff()
tab.insert(1, "bloco_acrescentado", ["criminal", "temporal", "espacial", "perfil"])
tab.to_csv(comum.PASTA / "ablacao.csv", index=False)

pd.DataFrame([dict(modelo=n, territoriais=", ".join(a for a in SENSIBILIDADE[n] if a in B["espacial"]), **metricas[n])
              for n in SENSIBILIDADE]).to_csv(comum.PASTA / "sensibilidade_metricas.csv", index=False)
longas = []
for n, c in contribs.items():
    # prefixo nas contribuições: as de bairro, aisp e risp teriam o nome das colunas de identificação
    c = c.rename(columns=lambda a: a if a == "valor_base" else "contrib_" + a)
    d = pd.concat([base.loc[c.index, ["bairro", "ano_alvo", "mes_alvo"]], c], axis=1)
    d.insert(0, "modelo", n)
    longas.append(d)
pd.concat(longas).to_csv(comum.PASTA / "contribuicoes_sensibilidade.csv", index=False)

importancia = pd.DataFrame({n: contribs[n][NAO_ESPACIAIS].abs().mean() for n in SENSIBILIDADE})
importancia.index.name = "atributo"
for n in SENSIBILIDADE:
    importancia["ordem_" + n] = importancia[n].rank(ascending=False, method="min").astype(int)
importancia.reset_index().to_csv(comum.PASTA / "sensibilidade_importancia.csv", index=False)
taus = []
for a, b in itertools.combinations(SENSIBILIDADE, 2):
    tau, _ = kendalltau(importancia[a], importancia[b], variant="b")
    taus.append(dict(par="%s x %s" % (a, b), tau_b=float(tau)))
    comum.registrar("tau-b %s x %s: %.6f" % (a, b, tau))
pd.DataFrame(taus).to_csv(comum.PASTA / "sensibilidade_tau_b.csv", index=False)
espacial = contribs["M4"][B["espacial"]].abs().mean()
peso = pd.DataFrame({"atributo": B["espacial"], "media_abs": espacial.to_numpy(),
                     "parte_da_soma": (espacial / espacial.sum()).to_numpy()})
peso.to_csv(comum.PASTA / "sensibilidade_peso_espacial.csv", index=False)
comum.registrar("peso espacial no M4 (média do valor absoluto e parte): %s"
                % "; ".join("%s %.6f (%.4f)" % r for r in peso.itertuples(index=False)))
comum.registrar("etapas 4 e 5: fim")
