"""Bloco 37, item 9: quadro com os 4 modelos da Parte 1 (composição antiga dos blocos) e os 4 da Parte 4 (composição
de 21/10), em 2023 (média das 12 dobras) e em 2024 (corte fixo). Só lê; grava comparacao_partes_1_e_4.csv nesta pasta,
sem sobrescrever.

- Parte 1, 2023: a média das 12 dobras sai das previsões que a Parte 1 gravou (previsoes_ablacao_sensibilidade.csv),
  dobra a dobra, com as métricas do comum.py; a Parte 1 gravou as métricas só nas 1.992 linhas juntas (ablacao.csv).
  As previsões voltam a float32 antes da conta, como na conferência da Parte 1: o CSV guarda o texto do float32.
- Parte 1, 2024: a ablação no corte fixo do teste da Parte 2 (parte2/ablacao_sensibilidade_2024.csv).
- Parte 4: ablacao_blocos2_2023.csv (linha media_das_12_dobras) e ablacao_blocos2_2024.csv.

Da raiz do repositório:
PYTHONUTF8=1 ./venv/Scripts/python.exe analises/medicoes_2026-09/rodada_2026-10-07/parte4/comparacao_parte4.py
"""
import sys

sys.dont_write_bytecode = True

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PARTE4 = Path(__file__).resolve().parent
RODADA = PARTE4.parent
sys.path.insert(0, str(RODADA))
import comum  # noqa: E402

METRICAS = ("mae", "rmse", "deviance")
ATRIBUTOS = {"antiga": {"M1": 16, "M2": 20, "M3": 23, "M4": 30}, "21/10": {"M1": 16, "M2": 24, "M3": 27, "M4": 30}}
SAIDA = PARTE4 / "comparacao_partes_1_e_4.csv"

linhas = []
p1 = pd.read_csv(RODADA / "previsoes_ablacao_sensibilidade.csv")
for m in ("M1", "M2", "M3", "M4"):
    d = p1[p1["modelo"] == m]
    assert len(d) == 1992
    dobras = [comum.metricas(g["alvo"], g["previsao"].to_numpy().astype(np.float32))
              for _, g in d.groupby("mes_alvo", sort=True)]
    assert len(dobras) == 12
    linhas.append({"parte": "Parte 1", "composicao": "antiga", "modelo": m, "atributos": ATRIBUTOS["antiga"][m],
                   "ano": 2023, "agregado": "media_das_12_dobras",
                   **{k: float(np.mean([x[k] for x in dobras])) for k in METRICAS}})
p2 = pd.read_csv(RODADA / "parte2" / "ablacao_sensibilidade_2024.csv").set_index("variante")
for m in ("M1", "M2", "M3", "M4"):
    assert p2.loc[m, "atributos"] == ATRIBUTOS["antiga"][m] and p2.loc[m, "linhas"] == 1992
    linhas.append({"parte": "Parte 2 (teste)", "composicao": "antiga", "modelo": m, "atributos": ATRIBUTOS["antiga"][m],
                   "ano": 2024, "agregado": "corte_fixo", **{k: float(p2.loc[m, k]) for k in METRICAS}})
v4 = pd.read_csv(PARTE4 / "ablacao_blocos2_2023.csv")
v4 = v4[v4["dobra"] == "media_das_12_dobras"].set_index("modelo")
t4 = pd.read_csv(PARTE4 / "ablacao_blocos2_2024.csv").set_index("modelo")
for ano, tab, agregado in ((2023, v4, "media_das_12_dobras"), (2024, t4, "corte_fixo")):
    for m in ("M1", "M2", "M3", "M4"):
        assert tab.loc[m, "atributos"] == ATRIBUTOS["21/10"][m]
        linhas.append({"parte": "Parte 4", "composicao": "21/10", "modelo": m, "atributos": ATRIBUTOS["21/10"][m],
                       "ano": ano, "agregado": agregado, **{k: float(tab.loc[m, k]) for k in METRICAS}})
tabela = pd.DataFrame(linhas)
with open(SAIDA, "x", encoding="utf-8", newline="") as f:
    tabela.to_csv(f, index=False)
print(tabela.to_string())
