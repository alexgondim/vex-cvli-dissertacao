"""Variante MD (29 atributos: os 27 não espaciais, aisp e risp, sem o bairro) nas 12 dobras de 2023, com as
contribuições, e no corte fixo de 2023, só as 3 métricas.

Roda no modo `ensaio`: as linhas de alvo em 2024 saem da base logo depois da leitura. Calcula a média do valor
absoluto da contribuição de cada atributo, a ordenação dos 27 não espaciais, o tau-b de Kendall contra o M4 e a MB e
o peso espacial (aisp e risp sobre os 29), com a mesma definição para o M4 e a MB. Lê a base por bairro e mês,
`resultados/sensibilidade_importancia.csv` e `resultados/sensibilidade_contribuicoes.csv`.

Grava em `resultados/`: `sensibilidade_md_dobras_2023.csv`, `sensibilidade_md_contribuicoes_2023.csv`,
`importancia_MD_2023.csv` e `sensibilidade_md_resumo_2023.json`.

Da raiz do repositório: python experimentos/sensibilidade_md_dobras_2023.py ensaio
"""
import json

import pandas as pd
from scipy.stats import kendalltau

import sensibilidade_comum_md as c

m = c.modo()
assert m == "ensaio", "este programa roda no modo ensaio"
c.registrar("ensaio: início; base lida sem as linhas de 2024")
base = c.ler_base(m)
treino, avaliadas = c.corte_fixo(base, 2023)
y23 = avaliadas[c.ALVO].to_numpy(float)
importancia_p1 = pd.read_csv(c.RESULTADOS / "sensibilidade_importancia.csv").set_index("atributo")

# MD nas 12 dobras, com contribuições
prev_md, contribs = c.comum.prever_dobras(base, c.MD, c.CONFIG, com_contribuicoes=True)
m_md = c.comum.metricas(prev_md[c.ALVO], prev_md["previsao"])
c.registrar("MD (29 atributos) nas 12 dobras: MAE %.6f, RMSE %.6f, deviance %.6f" % tuple(m_md.values()))
modelo_md = c.comum.estimador(c.CONFIG).fit(treino[c.MD], treino[c.ALVO].to_numpy(float))
m_md_cf = c.comum.metricas(y23, modelo_md.predict(avaliadas[c.MD]).astype(float))
c.registrar("MD no corte fixo de 2023: MAE %.6f, RMSE %.6f, deviance %.6f" % tuple(m_md_cf.values()))
c.c2.gravar_csv(pd.DataFrame([{"variante": "MD", "atributos": 29, "forma": "dobras_2023", "ano": 2023,
                                "linhas": 1992, **m_md},
                               {"variante": "MD", "atributos": 29, "forma": "corte_fixo", "ano": 2023,
                                "linhas": 1992, **m_md_cf}]), c.RESULTADOS / "sensibilidade_md_dobras_2023.csv")
longa = contribs.rename(columns=lambda a: a if a == "valor_base" else "contrib_" + a)
longa = pd.concat([c.identificacao(base.loc[longa.index]), longa], axis=1)
longa.insert(0, "modelo", "MD")
assert len(longa) == 1992
c.c2.gravar_csv(longa, c.RESULTADOS / "sensibilidade_md_contribuicoes_2023.csv")

medias = c.media_abs(contribs, c.MD)
imp = pd.DataFrame({"MD": medias})
imp.index.name = "atributo"
imp["ordem_MD_29"] = imp["MD"].rank(ascending=False, method="min").astype(int)
nao_esp = imp.loc[c.NAO_ESPACIAIS, "MD"]
imp["ordem_MD_27"] = nao_esp.rank(ascending=False, method="min").astype(int).reindex(imp.index)
for n in ("M4", "MB"):
    imp[n] = importancia_p1[n].reindex(imp.index)
    imp["ordem_%s_27" % n] = importancia_p1["ordem_" + n].reindex(imp.index)
c.c2.gravar_csv(imp.reset_index(), c.RESULTADOS / "importancia_MD_2023.csv")
taus = {}
for n in ("M4", "MB"):
    tau, _ = kendalltau(nao_esp.to_numpy(), importancia_p1.loc[c.NAO_ESPACIAIS, n].to_numpy(), variant="b")
    taus["MD x " + n] = float(tau)
    c.registrar("tau-b MD x %s nos 27 não espaciais: %.6f" % (n, tau))
peso = c.peso_espacial(medias, c.B["espacial"])
# a mesma definição para o M4 e a MB, a partir das contribuições das 12 dobras de 2023
cs = pd.read_csv(c.RESULTADOS / "sensibilidade_contribuicoes.csv")
pesos = {"MD": peso}
for n, nomes in (("M4", c.M4), ("MB", c.MB)):
    md = cs[cs["modelo"] == n][["contrib_" + a for a in nomes]].abs().mean()
    md.index = nomes
    pesos[n] = c.peso_espacial(md, c.B["espacial"])
c.registrar("peso espacial (aisp + risp sobre os 29) da MD: %.6f; mesma definição no M4 (bairro, aisp e "
            "risp sobre 30): %.6f; na MB (risp sobre 28): %.6f" % (pesos["MD"], pesos["M4"], pesos["MB"]))
c.c2.gravar_json({"ano": 2023, "forma": "dobras_2023", "metricas": m_md, "corte_fixo_2023": m_md_cf,
                  "ordem_27_MD": nao_esp.sort_values(ascending=False).index.tolist(),
                  "ordem_29_MD": medias.sort_values(ascending=False).index.tolist(),
                  "media_abs_29": {a: float(v) for a, v in medias.items()},
                  "tau_b": taus, "peso_espacial": pesos}, c.RESULTADOS / "sensibilidade_md_resumo_2023.json")
c.registrar("ensaio: fim; gravados sensibilidade_md_dobras_2023.csv, sensibilidade_md_contribuicoes_2023.csv, "
            "importancia_MD_2023.csv e sensibilidade_md_resumo_2023.json")
