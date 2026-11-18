"""Junta a validação no corte fixo e a na janela expansiva e confere a forma das tabelas.

Lê `validacao_metricas_corte_fixo.csv`, `validacao_metricas_janela.csv`, `validacao_metricas_mensais.csv`,
`validacao_previsoes_corte_fixo.csv`, `validacao_previsoes_janela.csv`, `validacao_nb_corte_fixo.json` e
`validacao_nb_janela.json`, da pasta do modo. Confere que cada linha das métricas anuais tem 1.992 linhas de alvo,
que cada linha das mensais tem 166 e o número de linhas das duas; no modo `ensaio`, confere também que a binomial
negativa converge no corte fixo. Se uma conferência falha, para sem gravar.

Grava `validacao_metricas.csv` (corte fixo e janela por ano), `previsoes_<modo>.csv` (corte fixo e janela, linha a
linha) e `validacao_binomial_negativa.json` (a convergência de cada ajuste da binomial negativa).

Da raiz do repositório: python experimentos/validacao_juntar_metricas.py teste
"""
import json
import sys

import pandas as pd

import validacao_comum as c

m = c.modo()
ano = c.ANO[m]
pasta = c.pasta_de_saida(m)
c.registrar("%s, junção e conferências: início" % m)
corte = pd.read_csv(pasta / "validacao_metricas_corte_fixo.csv")
janela = pd.read_csv(pasta / "validacao_metricas_janela.csv")
mensais = pd.read_csv(pasta / "validacao_metricas_mensais.csv")
metricas = pd.concat([corte, janela], ignore_index=True)[["forma", "ano", "metodo", "linhas", "mae", "rmse", "deviance"]]
problemas = []

if not (metricas["linhas"] == 1992).all():
    problemas.append("linha de metricas.csv sem 1.992 linhas de alvo")
if not (mensais["linhas"] == 166).all():
    problemas.append("linha de metricas_mensais.csv sem 166 linhas de alvo")
esperado = {"ensaio": (15, 24 * 5), "teste": (20, 36 * 5)}[m]
if (len(metricas), len(mensais)) != esperado:
    problemas.append("número de linhas de metricas.csv ou metricas_mensais.csv diferente de %s" % (esperado,))

nb_corte = json.loads((pasta / "validacao_nb_corte_fixo.json").read_text(encoding="utf-8"))
nb_janela = json.loads((pasta / "validacao_nb_janela.json").read_text(encoding="utf-8"))
if m == "ensaio" and not nb_corte["convergiu"]:
    problemas.append("a binomial negativa não convergiu no corte fixo")

for p in problemas:
    c.registrar("%s, conferência FALHOU: %s" % (m, p))
if problemas:
    sys.exit("conferências falharam; nada gravado")

previsoes = pd.concat([pd.read_csv(pasta / "validacao_previsoes_corte_fixo.csv"),
                       pd.read_csv(pasta / "validacao_previsoes_janela.csv")],
                      ignore_index=True)[["bairro", "ano_alvo", "mes_alvo", "forma", "metodo", "alvo", "previsao"]]
c.gravar_csv(metricas, pasta / "validacao_metricas.csv")
c.gravar_csv(previsoes, pasta / ("previsoes_%s.csv" % m))
regra = ("até %d recomeços a partir do último ponto; se ainda não convergir, a previsão sai do último ponto e o ajuste "
         "fica registrado como não convergido, sem olhar métrica nenhuma" % c.NB_RECOMECOS)
c.gravar_json({"criterio": "maior |gradiente| da log-verossimilhança média no ponto final abaixo de %g" % c.NB_TOL_GRADIENTE,
               "corte_fixo": {"ajustes": 1, "convergiram": int(nb_corte["convergiu"]), "ajuste": nb_corte},
               "janela": {"ajustes": len(nb_janela), "convergiram": sum(a["convergiu"] for a in nb_janela),
                          "ajustes_por_mes": nb_janela},
               "nao_convergidos": regra}, pasta / "validacao_binomial_negativa.json")
c.registrar("%s, conferências: 1.992 linhas em cada linha de validacao_metricas.csv; 166 em cada linha de "
            "validacao_metricas_mensais.csv%s" % (m, "; binomial negativa convergiu no corte fixo" if m == "ensaio" else ""))
c.registrar("%s, junção: gravados validacao_metricas.csv, previsoes_%s.csv e validacao_binomial_negativa.json" % (m, m))
