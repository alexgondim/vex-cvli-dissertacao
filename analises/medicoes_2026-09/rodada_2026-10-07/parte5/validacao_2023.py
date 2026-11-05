"""Itens 2 e 3 do bloco 42, só com alvos até 2023 (modo ensaio).

Item 2 primeiro: a MA (os 27 e o bairro) e a MD (os 27, aisp e risp), pela rotina das 12 dobras de 2023, têm de
repetir as 3 métricas do despacho até a 6ª casa. Se não repetirem, para. Registra também, sem ser critério de
parada, se as previsões e as métricas da MA repetem as da Parte 1 e se as métricas e as contribuições da MD repetem
as da Parte 3.

Item 3: ME, MF e MG nas 12 dobras, com as contribuições de cada linha pelo modelo da dobra dela; média do valor
absoluto por atributo; tau-b dos 27 não espaciais contra o M4 da Parte 1; peso espacial; atributo de maior média.

Grava conferencia_2023.json, sensibilidade_parte5_2023.csv, contribuicoes_ME_2023.csv, contribuicoes_MF_2023.csv,
contribuicoes_MG_2023.csv, importancia_parte5_2023.csv e resumo_parte5.json.
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import comum5 as c

CASAS = 6
DESPACHO = {"MA": {"mae": 0.576837, "rmse": 0.839612, "deviance": 1.008424},
            "MD": {"mae": 0.586980, "rmse": 0.850336, "deviance": 1.040701}}
CHAVE = ["bairro", "ano_alvo", "mes_alvo"]

m = c.modo()
assert m == "ensaio", "os itens 2 e 3 são do ensaio"
c.registrar("ensaio, itens 2 e 3: início; base lida sem as linhas de 2024")
base = c.c3.ler_base(m)
assert base["ano_alvo"].max() == 2023
importancia_p1 = pd.read_csv(c.RODADA / "sensibilidade_importancia.csv").set_index("atributo")


def repete(valores, esperados):
    return all(round(valores[k], CASAS) == esperados[k] for k in esperados)


def mesmas_linhas(a, b):
    """As mesmas linhas, na mesma ordem: bairro (como texto), ano e mês do alvo."""
    return bool((a["bairro"].astype(str).to_numpy() == b["bairro"].astype(str).to_numpy()).all()
                and (a[CHAVE[1:]].to_numpy() == b[CHAVE[1:]].to_numpy()).all())


# item 2: MA nas 12 dobras, contra o despacho e contra a Parte 1
prev_ma, _ = c.comum.prever_dobras(base, c.MA, c.CONFIG)
m_ma = c.comum.metricas(prev_ma[c.ALVO], prev_ma["previsao"])
p1 = pd.read_csv(c.RODADA / "previsoes_ablacao_sensibilidade.csv")
p1 = p1[p1["modelo"] == "MA"]
metricas_p1 = pd.read_csv(c.RODADA / "sensibilidade_metricas.csv").set_index("modelo")
ma = {**m_ma, "despacho": DESPACHO["MA"], "repete_o_despacho": repete(m_ma, DESPACHO["MA"]),
      # o XGBoost prevê em float32, e o CSV guarda o texto desse valor: a comparação volta a float32
      "previsoes_iguais_bit_a_bit_parte1": mesmas_linhas(p1, prev_ma) and bool(
          np.array_equal(p1["previsao"].to_numpy().astype(np.float32), prev_ma["previsao"].to_numpy())),
      "metricas_iguais_parte1": all(m_ma[k] == metricas_p1.loc["MA", k] for k in ("mae", "rmse", "deviance"))}
c.registrar("item 2, MA (%d atributos) nas 12 dobras: MAE %.6f, RMSE %.6f, deviance %.6f (%r, %r, %r); números do "
            "despacho repetidos: %s; previsões iguais bit a bit (float32) às da Parte 1: %s; 3 métricas iguais às da "
            "Parte 1: %s" % (len(c.MA), m_ma["mae"], m_ma["rmse"], m_ma["deviance"], m_ma["mae"], m_ma["rmse"],
                             m_ma["deviance"], ma["repete_o_despacho"], ma["previsoes_iguais_bit_a_bit_parte1"],
                             ma["metricas_iguais_parte1"]))

# item 2: MD nas 12 dobras, com as contribuições, contra o despacho e contra a Parte 3
prev_md, contribs_md = c.comum.prever_dobras(base, c.MD, c.CONFIG, com_contribuicoes=True)
m_md = c.comum.metricas(prev_md[c.ALVO], prev_md["previsao"])
p3 = pd.read_csv(c.RODADA / "parte3" / "sensibilidade_MD_2023.csv")
p3 = p3[p3["forma"] == "dobras_2023"].iloc[0]
cp3 = pd.read_csv(c.RODADA / "parte3" / "contribuicoes_MD_2023.csv")
colunas = ["contrib_" + a for a in c.MD] + ["valor_base"]
md = {**m_md, "despacho": DESPACHO["MD"], "repete_o_despacho": repete(m_md, DESPACHO["MD"]),
      "metricas_iguais_parte3": all(m_md[k] == p3[k] for k in ("mae", "rmse", "deviance")),
      "contribuicoes_iguais_bit_a_bit_parte3": mesmas_linhas(cp3, base.loc[contribs_md.index]) and bool(
          np.array_equal(cp3[colunas].to_numpy().astype(np.float32), contribs_md[c.MD + ["valor_base"]].to_numpy()))}
c.registrar("item 2, MD (%d atributos) nas 12 dobras: MAE %.6f, RMSE %.6f, deviance %.6f (%r, %r, %r); números do "
            "despacho repetidos: %s; 3 métricas iguais às da Parte 3: %s; contribuições iguais bit a bit (float32) às "
            "da Parte 3: %s" % (len(c.MD), m_md["mae"], m_md["rmse"], m_md["deviance"], m_md["mae"], m_md["rmse"],
                                m_md["deviance"], md["repete_o_despacho"], md["metricas_iguais_parte3"],
                                md["contribuicoes_iguais_bit_a_bit_parte3"]))
ok = ma["repete_o_despacho"] and md["repete_o_despacho"]
c.c2.gravar_json({"ok": bool(ok), "criterio": "as 3 métricas da MA e da MD iguais às do despacho até a 6ª casa",
                  "forma": "dobras_2023", "MA": ma, "MD": md}, c.PARTE5 / "conferencia_2023.json")
if not ok:
    c.registrar("PARADA: a MA ou a MD não repete os números do despacho")
    sys.exit(1)

# item 3: ME, MF e MG nas 12 dobras, com as contribuições
linhas, medias_por_variante, resumo = [], {}, {}
for v, nomes in c.VARIANTES.items():
    prev, contribs = c.comum.prever_dobras(base, nomes, c.CONFIG, com_contribuicoes=True)
    met = c.comum.metricas(prev[c.ALVO], prev["previsao"])
    esp = c.espaciais_de(nomes)
    # aditividade: soma das contribuições mais o valor base = margem = log da contagem esperada
    aditividade = float(np.abs(contribs.sum(axis=1).to_numpy(float) - np.log(prev["previsao"].to_numpy(float))).max())
    c.registrar("item 3, %s (%d atributos; espaciais: %s) nas 12 dobras: MAE %.6f, RMSE %.6f, deviance %.6f "
                "(%r, %r, %r); aditividade, maior |soma das contribuições e do valor base − log da previsão|: %.2e"
                % (v, len(nomes), ", ".join(esp) or "nenhum", met["mae"], met["rmse"], met["deviance"], met["mae"],
                   met["rmse"], met["deviance"], aditividade))
    linhas.append({"variante": v, "atributos": len(nomes), "espaciais": ", ".join(esp) or "nenhum",
                   "forma": "dobras_2023", "ano": 2023, "linhas": int(len(prev)), **met})
    longa = contribs.rename(columns=lambda a: a if a == "valor_base" else "contrib_" + a)
    longa = pd.concat([c.c3.identificacao(base.loc[longa.index]), longa], axis=1)
    longa.insert(0, "modelo", v)
    assert len(longa) == 1992
    c.c2.gravar_csv(longa, c.PARTE5 / ("contribuicoes_%s_2023.csv" % v))
    medias = c.c3.media_abs(contribs, nomes)
    medias_por_variante[v] = medias
    nao_esp = medias[c.NAO_ESPACIAIS]
    tau, _ = kendalltau(nao_esp.to_numpy(), importancia_p1.loc[c.NAO_ESPACIAIS, "M4"].to_numpy(), variant="b")
    peso = c.c3.peso_espacial(medias, c.ESPACIAIS)
    maior = medias.sort_values(ascending=False)
    c.registrar("item 3, %s: tau-b contra o M4 nos 27 não espaciais %.6f; peso espacial %.6f; maior média do valor "
                "absoluto: %s (%.6f)" % (v, tau, peso, maior.index[0], maior.iloc[0]))
    resumo[v] = {"atributos": len(nomes), "espaciais": esp, "metricas": met,
                 "tau_b_contra_M4": float(tau), "peso_espacial": peso,
                 "maior_media_abs": {"atributo": maior.index[0], "valor": float(maior.iloc[0])},
                 "ordem_27": nao_esp.sort_values(ascending=False).index.tolist(),
                 "ordem_todos": maior.index.tolist(),
                 "media_abs": {a: float(x) for a, x in medias.items()}}

c.c2.gravar_csv(pd.DataFrame(linhas), c.PARTE5 / "sensibilidade_parte5_2023.csv")
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
c.c2.gravar_csv(imp.reset_index(), c.PARTE5 / "importancia_parte5_2023.csv")
c.c2.gravar_json({"ano": 2023, "forma": "dobras_2023", "escala": "log da contagem esperada",
                  "tau_b": "kendalltau(variant='b') entre as médias do valor absoluto dos 27 não espaciais da variante "
                           "e as do M4 da Parte 1 (sensibilidade_importancia.csv)",
                  "peso_espacial": "soma das médias do valor absoluto dos atributos espaciais da variante dividida "
                                   "pela soma das médias de todos os atributos dela",
                  "variantes": resumo}, c.PARTE5 / "resumo_parte5.json")
c.registrar("ensaio, itens 2 e 3: fim; gravados conferencia_2023.json, sensibilidade_parte5_2023.csv, "
            "contribuicoes_ME_2023.csv, contribuicoes_MF_2023.csv, contribuicoes_MG_2023.csv, "
            "importancia_parte5_2023.csv e resumo_parte5.json")
