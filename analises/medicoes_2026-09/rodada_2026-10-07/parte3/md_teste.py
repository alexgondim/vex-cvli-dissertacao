"""Item 4 do bloco 31 (modo teste, 1 vez, depois do congelamento): a MD treinada nas 7.968 linhas de alvo de 2019
a 2022 e avaliada nas 1.992 linhas de 2024, com as 3 métricas, as contribuições nessas linhas e os resumos do
item 2 (média do valor absoluto, ordenação dos 27 não espaciais, tau-b contra o M4 do corte fixo e peso espacial).

Grava sensibilidade_MD_2024.csv, contribuicoes_MD_2024.csv, importancia_MD_2024.csv e resumo_MD.json (2023 e 2024).
"""
import json

import pandas as pd
from scipy.stats import kendalltau

import comum3 as c

m = c.modo()
assert m == "teste"
c.registrar("teste, item 4: início, MD no corte fixo avaliada nas linhas de alvo em 2024")
base = c.ler_base(m)
treino, avaliadas = c.corte_fixo(base, 2024)
y = avaliadas[c.ALVO].to_numpy(float)
modelo = c.comum.estimador(c.CONFIG).fit(treino[c.MD], treino[c.ALVO].to_numpy(float))
q = modelo.predict(avaliadas[c.MD]).astype(float)
metricas = c.comum.metricas(y, q)
c.registrar("item 4, MD (29 atributos) no corte fixo de 2024: MAE %.6f, RMSE %.6f, deviance %.6f" % tuple(metricas.values()))
c.c2.gravar_csv(pd.DataFrame([{"variante": "MD", "atributos": 29, "forma": "corte_fixo", "ano": 2024, "linhas": 1992,
                                **metricas}]), c.PARTE3 / "sensibilidade_MD_2024.csv")
contribs = pd.DataFrame(c.comum.contribuicoes(modelo, avaliadas[c.MD]), columns=c.MD + ["valor_base"],
                        index=avaliadas.index)
longa = contribs.rename(columns=lambda a: a if a == "valor_base" else "contrib_" + a)
longa = pd.concat([c.identificacao(avaliadas), longa], axis=1)
longa.insert(0, "modelo", "MD")
longa["alvo"] = y
longa["previsao"] = q
c.c2.gravar_csv(longa, c.PARTE3 / "contribuicoes_MD_2024.csv")

medias = c.media_abs(contribs, c.MD)
nao_esp = medias[c.NAO_ESPACIAIS]
# o M4 de 2024 é o do corte fixo da Parte 1, nas mesmas 1.992 linhas
cf = pd.read_csv(c.c2.ESTIMATIVAS)
cf = cf[cf["ano_alvo"] == 2024]
assert len(cf) == 1992
m4 = cf[["contrib_" + a for a in c.M4]].abs().mean()
m4.index = c.M4
imp = pd.DataFrame({"MD": medias})
imp.index.name = "atributo"
imp["ordem_MD_29"] = imp["MD"].rank(ascending=False, method="min").astype(int)
imp["ordem_MD_27"] = nao_esp.rank(ascending=False, method="min").astype(int).reindex(imp.index)
imp["M4"] = m4.reindex(imp.index)
imp["ordem_M4_27"] = m4[c.NAO_ESPACIAIS].rank(ascending=False, method="min").astype(int).reindex(imp.index)
c.c2.gravar_csv(imp.reset_index(), c.PARTE3 / "importancia_MD_2024.csv")
tau, _ = kendalltau(nao_esp.to_numpy(), m4[c.NAO_ESPACIAIS].to_numpy(), variant="b")
pesos = {"MD": c.peso_espacial(medias, c.B["espacial"]), "M4": c.peso_espacial(m4, c.B["espacial"])}
c.registrar("item 4, 2024: tau-b MD x M4 nos 27 não espaciais %.6f; peso espacial da MD %.6f, do M4 %.6f"
            % (tau, pesos["MD"], pesos["M4"]))
de_2023 = json.loads((c.PARTE3 / "resumo_MD_2023.json").read_text(encoding="utf-8"))
c.c2.gravar_json({"2023": de_2023,
                  "2024": {"forma": "corte_fixo", "metricas": metricas,
                           "ordem_27_MD": nao_esp.sort_values(ascending=False).index.tolist(),
                           "ordem_29_MD": medias.sort_values(ascending=False).index.tolist(),
                           "media_abs_29": {a: float(v) for a, v in medias.items()},
                           "tau_b": {"MD x M4": float(tau), "MD x MB": "sem contribuições da MB em 2024"},
                           "peso_espacial": pesos}}, c.PARTE3 / "resumo_MD.json")
c.registrar("teste, item 4: fim; gravados sensibilidade_MD_2024.csv, contribuicoes_MD_2024.csv, "
            "importancia_MD_2024.csv e resumo_MD.json")
