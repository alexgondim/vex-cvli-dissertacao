"""Itens 2 e 3 do bloco 31, só com alvos até 2023 (modo ensaio).

Item 3 primeiro: a MB pelas 2 rotinas (12 dobras de 2023, como a Parte 1; corte fixo avaliado em 2023, como a
Parte 2) tem de repetir os números fixados no despacho. Se não repetir, para.

Item 2: a MD (29 atributos, sem o bairro) nas 12 dobras, com as contribuições; média do valor absoluto por
atributo, ordenação dos 27 não espaciais, tau-b contra o M4 e a MB (médias da Parte 1) e peso espacial. Acréscimo
declarado no protocolo: a MD no corte fixo de 2023, só as 3 métricas.

Grava conferencia_MB.json, sensibilidade_MD_2023.csv, contribuicoes_MD_2023.csv, importancia_MD_2023.csv e
resumo_MD_2023.json.
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import comum3 as c

CASAS = 6
DESPACHO = {"dobras": {"deviance": 1.044035},
            "corte_fixo": {"mae": 0.587315, "rmse": 0.852285, "deviance": 1.048184}}

m = c.modo()
assert m == "ensaio", "os itens 2 e 3 são do ensaio"
c.registrar("ensaio, itens 3 e 2: início; base lida sem as linhas de 2024")
base = c.ler_base(m)
treino, avaliadas = c.corte_fixo(base, 2023)
y23 = avaliadas[c.ALVO].to_numpy(float)
previsoes_p1 = pd.read_csv(c.RODADA / "previsoes_ablacao_sensibilidade.csv")
metricas_p1 = pd.read_csv(c.RODADA / "sensibilidade_metricas.csv").set_index("modelo")
importancia_p1 = pd.read_csv(c.RODADA / "sensibilidade_importancia.csv").set_index("atributo")


def repete(valores, esperados):
    return all(round(valores[k], CASAS) == esperados[k] for k in esperados)


# item 3: MB nas 12 dobras (rotina da Parte 1)
prev_mb, _ = c.comum.prever_dobras(base, c.MB, c.CONFIG)
m_dobras = c.comum.metricas(prev_mb[c.ALVO], prev_mb["previsao"])
p1 = previsoes_p1[previsoes_p1["modelo"] == "MB"]["previsao"].to_numpy().astype(np.float32)
iguais_bit = bool(np.array_equal(p1, prev_mb["previsao"].to_numpy()))
iguais_p1 = all(m_dobras[k] == metricas_p1.loc["MB", k] for k in ("mae", "rmse", "deviance"))
ok_dobras = repete(m_dobras, DESPACHO["dobras"]) and iguais_bit and iguais_p1
c.registrar("item 3, MB nas 12 dobras: MAE %.6f, RMSE %.6f, deviance %.6f; deviance do despacho repetida: %s; "
            "previsões iguais bit a bit (float32) às da Parte 1: %s; 3 métricas iguais às da Parte 1: %s"
            % (m_dobras["mae"], m_dobras["rmse"], m_dobras["deviance"], repete(m_dobras, DESPACHO["dobras"]),
               iguais_bit, iguais_p1))
# item 3: MB no corte fixo de 2023 (rotina da Parte 2)
modelo_mb = c.comum.estimador(c.CONFIG).fit(treino[c.MB], treino[c.ALVO].to_numpy(float))
m_cf = c.comum.metricas(y23, modelo_mb.predict(avaliadas[c.MB]).astype(float))
ok_cf = repete(m_cf, DESPACHO["corte_fixo"])
c.registrar("item 3, MB no corte fixo de 2023: MAE %.6f, RMSE %.6f, deviance %.6f; números do despacho repetidos: %s"
            % (m_cf["mae"], m_cf["rmse"], m_cf["deviance"], ok_cf))
c.c2.gravar_json({"ok": bool(ok_dobras and ok_cf), "dobras_2023": {**m_dobras, "despacho": DESPACHO["dobras"],
                  "previsoes_iguais_bit_a_bit_parte1": iguais_bit, "metricas_iguais_parte1": iguais_p1},
                  "corte_fixo_2023": {**m_cf, "despacho": DESPACHO["corte_fixo"]}}, c.PARTE3 / "conferencia_MB.json")
if not (ok_dobras and ok_cf):
    c.registrar("PARADA: a MB não repete os números do despacho")
    sys.exit(1)

# item 2: MD nas 12 dobras, com contribuições
prev_md, contribs = c.comum.prever_dobras(base, c.MD, c.CONFIG, com_contribuicoes=True)
m_md = c.comum.metricas(prev_md[c.ALVO], prev_md["previsao"])
c.registrar("item 2, MD (29 atributos) nas 12 dobras: MAE %.6f, RMSE %.6f, deviance %.6f" % tuple(m_md.values()))
modelo_md = c.comum.estimador(c.CONFIG).fit(treino[c.MD], treino[c.ALVO].to_numpy(float))
m_md_cf = c.comum.metricas(y23, modelo_md.predict(avaliadas[c.MD]).astype(float))
c.registrar("acréscimo, MD no corte fixo de 2023: MAE %.6f, RMSE %.6f, deviance %.6f" % tuple(m_md_cf.values()))
c.c2.gravar_csv(pd.DataFrame([{"variante": "MD", "atributos": 29, "forma": "dobras_2023", "ano": 2023,
                                "linhas": 1992, **m_md},
                               {"variante": "MD", "atributos": 29, "forma": "corte_fixo", "ano": 2023,
                                "linhas": 1992, **m_md_cf}]), c.PARTE3 / "sensibilidade_MD_2023.csv")
longa = contribs.rename(columns=lambda a: a if a == "valor_base" else "contrib_" + a)
longa = pd.concat([c.identificacao(base.loc[longa.index]), longa], axis=1)
longa.insert(0, "modelo", "MD")
assert len(longa) == 1992
c.c2.gravar_csv(longa, c.PARTE3 / "contribuicoes_MD_2023.csv")

medias = c.media_abs(contribs, c.MD)
imp = pd.DataFrame({"MD": medias})
imp.index.name = "atributo"
imp["ordem_MD_29"] = imp["MD"].rank(ascending=False, method="min").astype(int)
nao_esp = imp.loc[c.NAO_ESPACIAIS, "MD"]
imp["ordem_MD_27"] = nao_esp.rank(ascending=False, method="min").astype(int).reindex(imp.index)
for n in ("M4", "MB"):
    imp[n] = importancia_p1[n].reindex(imp.index)
    imp["ordem_%s_27" % n] = importancia_p1["ordem_" + n].reindex(imp.index)
c.c2.gravar_csv(imp.reset_index(), c.PARTE3 / "importancia_MD_2023.csv")
taus = {}
for n in ("M4", "MB"):
    tau, _ = kendalltau(nao_esp.to_numpy(), importancia_p1.loc[c.NAO_ESPACIAIS, n].to_numpy(), variant="b")
    taus["MD x " + n] = float(tau)
    c.registrar("item 2, tau-b MD x %s nos 27 não espaciais: %.6f" % (n, tau))
peso = c.peso_espacial(medias, c.B["espacial"])
# a mesma definição para o M4 e a MB da Parte 1, a partir das contribuições gravadas lá
cs = pd.read_csv(c.RODADA / "contribuicoes_sensibilidade.csv")
pesos = {"MD": peso}
for n, nomes in (("M4", c.M4), ("MB", c.MB)):
    md = cs[cs["modelo"] == n][["contrib_" + a for a in nomes]].abs().mean()
    md.index = nomes
    pesos[n] = c.peso_espacial(md, c.B["espacial"])
c.registrar("item 2, peso espacial (aisp + risp sobre os 29) da MD: %.6f; mesma definição no M4 (bairro, aisp e "
            "risp sobre 30): %.6f; na MB (risp sobre 28): %.6f" % (pesos["MD"], pesos["M4"], pesos["MB"]))
c.c2.gravar_json({"ano": 2023, "forma": "dobras_2023", "metricas": m_md, "corte_fixo_2023": m_md_cf,
                  "ordem_27_MD": nao_esp.sort_values(ascending=False).index.tolist(),
                  "ordem_29_MD": medias.sort_values(ascending=False).index.tolist(),
                  "media_abs_29": {a: float(v) for a, v in medias.items()},
                  "tau_b": taus, "peso_espacial": pesos}, c.PARTE3 / "resumo_MD_2023.json")
c.registrar("ensaio, itens 3 e 2: fim; gravados conferencia_MB.json, sensibilidade_MD_2023.csv, "
            "contribuicoes_MD_2023.csv, importancia_MD_2023.csv e resumo_MD_2023.json")
