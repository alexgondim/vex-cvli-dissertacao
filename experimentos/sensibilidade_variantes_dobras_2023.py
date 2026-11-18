"""Variantes ME (bairro e aisp), MF (só aisp) e MG (nenhum atributo espacial) nas 12 dobras de 2023, com as
contribuições de cada linha pelo modelo da dobra dela.

Roda no modo `ensaio`: as linhas de alvo em 2024 saem da base logo depois da leitura. Para cada variante: as 3
métricas, a média do valor absoluto por atributo, o tau-b de Kendall dos 27 não espaciais contra o M4, o peso
espacial e o atributo de maior média. Lê a base por bairro e mês e `resultados/sensibilidade_importancia.csv` (as
médias do M4).

Grava em `resultados/`: `sensibilidade_variantes_dobras_2023.csv`, `sensibilidade_me_contribuicoes_2023.csv`,
`sensibilidade_mf_contribuicoes_2023.csv`, `sensibilidade_mg_contribuicoes_2023.csv`,
`sensibilidade_variantes_importancia_2023.csv` e `sensibilidade_variantes_resumo_2023.json`.

Da raiz do repositório: python experimentos/sensibilidade_variantes_dobras_2023.py ensaio
"""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import sensibilidade_comum_variantes as c

CHAVE = ["bairro", "ano_alvo", "mes_alvo"]

m = c.modo()
assert m == "ensaio", "este programa roda no modo ensaio"
c.registrar("ensaio: início; base lida sem as linhas de 2024")
base = c.c3.ler_base(m)
assert base["ano_alvo"].max() == 2023
importancia_p1 = pd.read_csv(c.RESULTADOS / "sensibilidade_importancia.csv").set_index("atributo")


# ME, MF e MG nas 12 dobras, com as contribuições
linhas, medias_por_variante, resumo = [], {}, {}
for v, nomes in c.VARIANTES.items():
    prev, contribs = c.comum.prever_dobras(base, nomes, c.CONFIG, com_contribuicoes=True)
    met = c.comum.metricas(prev[c.ALVO], prev["previsao"])
    esp = c.espaciais_de(nomes)
    # aditividade: soma das contribuições mais o valor base = margem = log da contagem esperada
    aditividade = float(np.abs(contribs.sum(axis=1).to_numpy(float) - np.log(prev["previsao"].to_numpy(float))).max())
    c.registrar("%s (%d atributos; espaciais: %s) nas 12 dobras: MAE %.6f, RMSE %.6f, deviance %.6f "
                "(%r, %r, %r); aditividade, maior |soma das contribuições e do valor base − log da previsão|: %.2e"
                % (v, len(nomes), ", ".join(esp) or "nenhum", met["mae"], met["rmse"], met["deviance"], met["mae"],
                   met["rmse"], met["deviance"], aditividade))
    linhas.append({"variante": v, "atributos": len(nomes), "espaciais": ", ".join(esp) or "nenhum",
                   "forma": "dobras_2023", "ano": 2023, "linhas": int(len(prev)), **met})
    longa = contribs.rename(columns=lambda a: a if a == "valor_base" else "contrib_" + a)
    longa = pd.concat([c.c3.identificacao(base.loc[longa.index]), longa], axis=1)
    longa.insert(0, "modelo", v)
    assert len(longa) == 1992
    c.c2.gravar_csv(longa, c.RESULTADOS / ("sensibilidade_%s_contribuicoes_2023.csv" % v.lower()))
    medias = c.c3.media_abs(contribs, nomes)
    medias_por_variante[v] = medias
    nao_esp = medias[c.NAO_ESPACIAIS]
    tau, _ = kendalltau(nao_esp.to_numpy(), importancia_p1.loc[c.NAO_ESPACIAIS, "M4"].to_numpy(), variant="b")
    peso = c.c3.peso_espacial(medias, c.ESPACIAIS)
    maior = medias.sort_values(ascending=False)
    c.registrar("%s: tau-b contra o M4 nos 27 não espaciais %.6f; peso espacial %.6f; maior média do valor "
                "absoluto: %s (%.6f)" % (v, tau, peso, maior.index[0], maior.iloc[0]))
    resumo[v] = {"atributos": len(nomes), "espaciais": esp, "metricas": met,
                 "tau_b_contra_M4": float(tau), "peso_espacial": peso,
                 "maior_media_abs": {"atributo": maior.index[0], "valor": float(maior.iloc[0])},
                 "ordem_27": nao_esp.sort_values(ascending=False).index.tolist(),
                 "ordem_todos": maior.index.tolist(),
                 "media_abs": {a: float(x) for a, x in medias.items()}}

c.c2.gravar_csv(pd.DataFrame(linhas), c.RESULTADOS / "sensibilidade_variantes_dobras_2023.csv")
imp = pd.DataFrame(index=pd.Index(c.VARIANTES["ME"], name="atributo"))
for v, nomes in c.VARIANTES.items():
    medias = medias_por_variante[v]
    imp[v] = medias.reindex(imp.index)
    imp["ordem_%s_%d" % (v, len(nomes))] = medias.rank(ascending=False, method="min").astype(int).reindex(imp.index)
    if len(nomes) > 27:
        imp["ordem_%s_27" % v] = (medias[c.NAO_ESPACIAIS].rank(ascending=False, method="min").astype(int)
                                  .reindex(imp.index))
imp["M4"] = importancia_p1["M4"].reindex(imp.index)
imp["ordem_M4_27"] = importancia_p1["ordem_M4"].reindex(imp.index)
c.c2.gravar_csv(imp.reset_index(), c.RESULTADOS / "sensibilidade_variantes_importancia_2023.csv")
c.c2.gravar_json({"ano": 2023, "forma": "dobras_2023", "escala": "log da contagem esperada",
                  "tau_b": "kendalltau(variant='b') entre as médias do valor absoluto dos 27 não espaciais da variante "
                           "e as do M4 do sensibilidade_dobras_2023.py (sensibilidade_importancia.csv)",
                  "peso_espacial": "soma das médias do valor absoluto dos atributos espaciais da variante dividida "
                                   "pela soma das médias de todos os atributos dela",
                  "variantes": resumo}, c.RESULTADOS / "sensibilidade_variantes_resumo_2023.json")
c.registrar("ensaio: fim; gravados sensibilidade_variantes_dobras_2023.csv, sensibilidade_me_contribuicoes_2023.csv, "
            "sensibilidade_mf_contribuicoes_2023.csv, sensibilidade_mg_contribuicoes_2023.csv, "
            "sensibilidade_variantes_importancia_2023.csv e sensibilidade_variantes_resumo_2023.json")
