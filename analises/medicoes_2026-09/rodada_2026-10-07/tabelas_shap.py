"""Item 10 do bloco 26: tabelas dos 5 usos do SHAP, tiradas das contribuições do modelo do corte fixo, sem modelo
novo. Escala: log da contagem esperada. Nenhuma tabela traz o alvo de 2024.

a. global: média do valor absoluto de cada atributo nas 1.992 linhas de 2024, com a ordem dos 30;
b. dependência: valor e contribuição das 3 proporções de turno e da tendência nas linhas de 2024, com a RISP e a
   soma das interações espaciais; nas proporções, sem as linhas de janela vazia;
c. local: Itapuã, mês-alvo junho de 2024;
d. entre anos: contribuições das linhas de alvo em 2019, 2021 e 2024;
e. espacial: média do valor absoluto de cada atributo e do total (soma dos valores absolutos dos 30 em cada
   linha), por bairro e por RISP, nas linhas de 2024.
"""
import numpy as np
import pandas as pd

import comum

DEPENDENCIA = ["prop_madrugada_hist12", "prop_tarde_hist12", "prop_noite_hist12", "cvli_trend"]
CASO = ("Itapuã", 2024, 6)
nomes = comum.atributos()
bloco_de = {a: b for b, lista in comum.BLOCOS.items() for a in lista}
base = comum.ler_base().reset_index(drop=True)
c = pd.read_csv(comum.PASTA / "contribuicoes_corte_fixo.csv")
assert (c[["bairro", "ano_alvo", "mes_alvo"]].to_numpy() == base[["bairro", "ano_alvo", "mes_alvo"]].astype(object).to_numpy()).all()
phi = c[["contrib_" + a for a in nomes]].set_axis(nomes, axis=1)
em_2024 = c["ano_alvo"] == 2024
comum.registrar("etapa 9, tabelas dos 5 usos do SHAP")

media = phi[em_2024].abs().mean().sort_values(ascending=False)
pd.DataFrame({"ordem": range(1, 31), "atributo": media.index, "bloco": [bloco_de[a] for a in media.index],
              "media_abs": media.to_numpy()}).to_csv(comum.PASTA / "shap_a_global_2024.csv", index=False)
comum.registrar("global: os 10 primeiros %s" % "; ".join("%s %.6f" % (a, v) for a, v in media.head(10).items()))

inter = pd.read_csv(comum.PASTA / "interacoes_2024.csv")
assert len(inter) == 1992 and (inter[["bairro", "mes_alvo"]].to_numpy() == c.loc[em_2024, ["bairro", "mes_alvo"]].to_numpy()).all()
partes = []
for a in DEPENDENCIA:
    d = pd.DataFrame({"atributo": a, "bairro": c.loc[em_2024, "bairro"].to_numpy(),
                      "ano_alvo": 2024, "mes_alvo": c.loc[em_2024, "mes_alvo"].to_numpy(),
                      "risp": base.loc[em_2024, "risp"].astype(str).to_numpy(), "valor": base.loc[em_2024, a].to_numpy(),
                      "contribuicao": phi.loc[em_2024, a].to_numpy(), "soma_interacoes_espaciais": inter[a + "__soma"].to_numpy()})
    if a != "cvli_trend":
        d = d[d["valor"].notna()]
    partes.append(d)
    comum.registrar("dependência %s: %d linhas" % (a, len(d)))
pd.concat(partes).to_csv(comum.PASTA / "shap_b_dependencia_2024.csv", index=False)

k = c.index[(c["bairro"] == CASO[0]) & (c["ano_alvo"] == CASO[1]) & (c["mes_alvo"] == CASO[2])]
assert len(k) == 1
k = k[0]
local = pd.DataFrame({"item": nomes, "valor": [base.loc[k, a] if a not in comum.CATEGORICAS else str(base.loc[k, a]) for a in nomes],
                      "contribuicao": phi.loc[k, nomes].to_numpy()})
local["ordem_abs"] = local["contribuicao"].abs().rank(ascending=False, method="first").astype(int)
resumo = pd.DataFrame({"item": ["valor de base", "margem", "estimativa (registros esperados)", "n_vitimas_hist12"],
                       "valor": [c.loc[k, "valor_base"], c.loc[k, "margem"], c.loc[k, "estimativa"], base.loc[k, "n_vitimas_hist12"]]})
pd.concat([local, resumo]).to_csv(comum.PASTA / "shap_c_local_itapua_2024_06.csv", index=False)
assert abs(c.loc[k, "valor_base"] + local["contribuicao"].sum() - c.loc[k, "margem"]) < 1e-4
comum.registrar("local Itapuã jun/2024: valor de base %.6f, margem %.6f, estimativa %.6f registros esperados, n_vitimas_hist12 %d"
                % (c.loc[k, "valor_base"], c.loc[k, "margem"], c.loc[k, "estimativa"], base.loc[k, "n_vitimas_hist12"]))

anos = c["ano_alvo"].isin([2019, 2021, 2024])
c.loc[anos, ["bairro", "ano_alvo", "mes_alvo", "divisao"] + ["contrib_" + a for a in nomes]].to_csv(
    comum.PASTA / "shap_d_entre_anos.csv", index=False)

# prefixo nas médias: bairro e risp, como atributos, teriam o nome das colunas de agrupamento
absoluto = phi[em_2024].abs().assign(total=phi[em_2024].abs().sum(axis=1)).add_prefix("media_abs_")
chaves = pd.DataFrame({"bairro": c.loc[em_2024, "bairro"], "risp": base.loc[em_2024, "risp"].astype(str)})
por_bairro = pd.concat([chaves, absoluto], axis=1).groupby(["bairro", "risp"]).mean().reset_index()
por_risp = pd.concat([chaves[["risp"]], absoluto], axis=1).groupby("risp").mean().reset_index()
assert len(por_bairro) == 166 and len(por_risp) == 3
por_bairro.to_csv(comum.PASTA / "shap_e_espacial_bairro_2024.csv", index=False)
por_risp.to_csv(comum.PASTA / "shap_e_espacial_risp_2024.csv", index=False)
comum.registrar("espacial: total médio por RISP %s" % "; ".join("%s %.6f" % r for r in por_risp[["risp", "media_abs_total"]].itertuples(index=False)))
comum.registrar("etapa 9: fim")
