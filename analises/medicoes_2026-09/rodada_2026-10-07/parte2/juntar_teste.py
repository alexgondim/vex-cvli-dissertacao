"""Junta as 3 formas e faz as conferências. Grava metricas.csv (corte fixo e janela por ano), previsoes_<modo>.csv
(corte fixo e janela, linha a linha) e binomial_negativa.json.

Conferências, no log: o M4 da ablação repete até a 6ª casa as métricas do modelo no corte fixo; toda linha de
metricas.csv tem 1.992 linhas de alvo e toda linha de metricas_mensais.csv, 166. No ensaio, os números da
conferência do despacho repetem até a 6ª casa e a binomial negativa converge no corte fixo (grava
ensaio/conferencia_ensaio.json, que o congelamento exige). No teste, as métricas da janela em 2022 e 2023 repetem as
do ensaio até a 6ª casa.
"""
import json
import sys

import pandas as pd

import comum2 as c

# números da conferência, calculados só com alvos de 2022 e 2023 (despacho do bloco 30, item 2)
CONFERENCIA = [
    ("corte_fixo", 2023, "modelo", 0.581558, 0.845405, 1.017706),
    ("corte_fixo", 2023, "persistencia", 0.653614, 1.167670, 2.799974),
    ("corte_fixo", 2023, "media_movel_12", 0.564759, 0.868799, 1.109814),
    ("corte_fixo", 2023, "media_sazonal", 0.604920, 0.938864, 1.405426),
    ("janela", 2022, "modelo", 0.599385, 0.887553, 1.015303),
    ("janela", 2023, "modelo", 0.575637, 0.839071, 1.007328),
    ("janela", 2022, "persistencia", 0.680723, 1.192345, 2.756077),
    ("janela", 2022, "media_movel_12", 0.601531, 0.920225, 1.104318),
    ("janela", 2022, "media_sazonal", 0.612952, 0.975059, 1.439375),
    ("janela", 2023, "persistencia", 0.653614, 1.167670, 2.799974),
    ("janela", 2023, "media_movel_12", 0.564759, 0.868799, 1.109814),
    ("janela", 2023, "media_sazonal", 0.604920, 0.938864, 1.405426),
]
CONFERENCIA_ABLACAO = {"M1": (0.588104, 0.853168, 1.050205), "M2": (0.586847, 0.852032, 1.048503),
                       "M3": (0.579338, 0.843056, 1.015304), "M4": (0.581558, 0.845405, 1.017706),
                       "MA": (0.582672, 0.845783, 1.018855), "MB": (0.587315, 0.852285, 1.048184),
                       "MC": (0.583498, 0.846609, 1.020048)}
METRICAS = ("mae", "rmse", "deviance")


def seis(x):
    return round(float(x), 6)


m = c.modo()
ano = c.ANO[m]
pasta = c.pasta_de_saida(m)
c.registrar("%s, junção e conferências: início" % m)
corte = pd.read_csv(pasta / "metricas_corte_fixo.csv")
janela = pd.read_csv(pasta / "metricas_janela.csv")
mensais = pd.read_csv(pasta / "metricas_mensais.csv")
ablacao = pd.read_csv(pasta / ("ablacao_sensibilidade_%d.csv" % ano)).set_index("variante")
metricas = pd.concat([corte, janela], ignore_index=True)[["forma", "ano", "metodo", "linhas", "mae", "rmse", "deviance"]]
problemas = []

modelo = corte.set_index("metodo").loc["modelo"]
if any(seis(ablacao.loc["M4", k]) != seis(modelo[k]) for k in METRICAS):
    problemas.append("o M4 da ablação não repete o modelo do corte fixo")
if not (metricas["linhas"] == 1992).all():
    problemas.append("linha de metricas.csv sem 1.992 linhas de alvo")
if not (mensais["linhas"] == 166).all():
    problemas.append("linha de metricas_mensais.csv sem 166 linhas de alvo")
esperado = {"ensaio": (15, 24 * 5), "teste": (20, 36 * 5)}[m]
if (len(metricas), len(mensais)) != esperado:
    problemas.append("número de linhas de metricas.csv ou metricas_mensais.csv diferente de %s" % (esperado,))

nb_corte = json.loads((pasta / "nb_corte_fixo.json").read_text(encoding="utf-8"))
nb_janela = json.loads((pasta / "nb_janela.json").read_text(encoding="utf-8"))
indice = metricas.set_index(["forma", "ano", "metodo"])
if m == "ensaio":
    for forma, a, metodo, *valores in CONFERENCIA:
        obtido = [seis(indice.loc[(forma, a, metodo), k]) for k in METRICAS]
        if obtido != valores:
            problemas.append("conferência: %s %d %s deu %s, esperado %s" % (forma, a, metodo, obtido, valores))
    for variante, valores in CONFERENCIA_ABLACAO.items():
        obtido = [seis(ablacao.loc[variante, k]) for k in METRICAS]
        if obtido != list(valores):
            problemas.append("conferência: %s deu %s, esperado %s" % (variante, obtido, list(valores)))
    if not nb_corte["convergiu"]:
        problemas.append("a binomial negativa não convergiu no corte fixo")
else:
    ensaio = pd.read_csv(c.PASTA_ENSAIO / "metricas_janela.csv").set_index(["forma", "ano", "metodo"])
    for chave in ensaio.index:
        if [seis(ensaio.loc[chave, k]) for k in METRICAS] != [seis(indice.loc[chave, k]) for k in METRICAS]:
            problemas.append("janela %s não repete o ensaio" % (chave,))

for p in problemas:
    c.registrar("%s, conferência FALHOU: %s" % (m, p))
if problemas:
    sys.exit("conferências falharam; nada gravado")

previsoes = pd.concat([pd.read_csv(pasta / "previsoes_corte_fixo.csv"), pd.read_csv(pasta / "previsoes_janela.csv")],
                      ignore_index=True)[["bairro", "ano_alvo", "mes_alvo", "forma", "metodo", "alvo", "previsao"]]
c.gravar_csv(metricas, pasta / "metricas.csv")
c.gravar_csv(previsoes, pasta / ("previsoes_%s.csv" % m))
regra = ("até %d recomeços a partir do último ponto; se ainda não convergir, a previsão sai do último ponto e o ajuste "
         "fica registrado como não convergido, sem olhar métrica nenhuma" % c.NB_RECOMECOS)
c.gravar_json({"criterio": "maior |gradiente| da log-verossimilhança média no ponto final abaixo de %g" % c.NB_TOL_GRADIENTE,
               "corte_fixo": {"ajustes": 1, "convergiram": int(nb_corte["convergiu"]), "ajuste": nb_corte},
               "janela": {"ajustes": len(nb_janela), "convergiram": sum(a["convergiu"] for a in nb_janela),
                          "ajustes_por_mes": nb_janela},
               "nao_convergidos": regra}, pasta / "binomial_negativa.json")
if m == "ensaio":
    c.gravar_json({"ok": True, "ensaio": "números da conferência repetidos até a 6ª casa; binomial negativa convergiu "
                                         "no corte fixo"}, pasta / "conferencia_ensaio.json")
c.registrar("%s, conferências: M4 da ablação = modelo do corte fixo até a 6ª casa; 1.992 linhas em cada linha de "
            "metricas.csv; 166 em cada linha de metricas_mensais.csv%s" % (
                m, "; números da conferência do despacho repetidos até a 6ª casa; binomial negativa convergiu no "
                   "corte fixo" if m == "ensaio" else "; janela de 2022 e 2023 igual à do ensaio até a 6ª casa"))
c.registrar("%s, junção: gravados metricas.csv, previsoes_%s.csv e binomial_negativa.json" % (m, m))
