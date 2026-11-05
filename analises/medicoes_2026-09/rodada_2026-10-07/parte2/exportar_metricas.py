"""Arquivos do teste para o painel: copia metricas.csv e metricas_mensais.csv, sem mudar nada, para
painel_mensal/dados_modelo/, e grava ali teste.json com a data e a hora do CONGELADO_parte2.tsv, o sha256 dele e a
convergência da binomial negativa. Os arquivos que já estão em dados_modelo/ não mudam."""
import json
import shutil

import pandas as pd

import comum2 as c

DESTINO = c.RODADA.parents[2] / "painel_mensal" / "dados_modelo"
assert c.CONGELADO.exists() and (c.PARTE2 / "metricas.csv").exists()
antes = {p.name: c.sha256(p) for p in DESTINO.iterdir()}
for nome in ("metricas.csv", "metricas_mensais.csv", "teste.json"):
    assert not (DESTINO / nome).exists(), nome
for nome in ("metricas.csv", "metricas_mensais.csv"):
    shutil.copyfile(c.PARTE2 / nome, DESTINO / nome)
    assert c.sha256(DESTINO / nome) == c.sha256(c.PARTE2 / nome)
congelado = pd.read_csv(c.CONGELADO, sep="\t")
nb = json.loads((c.PARTE2 / "binomial_negativa.json").read_text(encoding="utf-8"))
with open(DESTINO / "teste.json", "x", encoding="utf-8") as f:
    json.dump({"congelado_data_hora": congelado["data_hora"].iloc[0], "congelado_sha256": c.sha256(c.CONGELADO),
               "binomial_negativa": {"corte_fixo_convergiu": bool(nb["corte_fixo"]["convergiram"] == 1),
                                     "janela_convergiram": int(nb["janela"]["convergiram"]),
                                     "janela_ajustes": int(nb["janela"]["ajustes"])}}, f, ensure_ascii=False, indent=1)
assert {p.name: c.sha256(p) for p in DESTINO.iterdir() if p.name in antes} == antes
c.registrar("exportação: metricas.csv e metricas_mensais.csv copiados para painel_mensal/dados_modelo/, e teste.json "
            "gravado ali; explicacoes.csv e rodada.json sem mudança")
