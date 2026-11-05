"""Números do capítulo da Parte 3 da rodada de 07/10 (bloco 31): a variante MD e a LSTM de viabilidade.

Lê as saídas copiadas de rodada_2026-10-07/parte3/ e, em entradas/, as cópias dos arquivos da Parte 1 e da Parte 2
que entram na tabela comparativa (M4, MB e média móvel de 12 meses em 2023 e 2024). Grava numeros_capitulo.csv (um
número por linha, com arquivo, linha ou chave e coluna) e numeros_capitulo.md (quadros, com a tabela comparativa do
item 12). Recusa sobrescrever. Executar da raiz do projeto."""
import json
from pathlib import Path

import pandas as pd

PASTA = Path(__file__).resolve().parent
E = PASTA / "entradas"
METRICAS = ("mae", "rmse", "deviance")
NOME = {"mae": "MAE", "rmse": "RMSE", "deviance": "deviance de Poisson"}
numeros = []


def casas(v):
    return f"{v:.6f}".replace(".", ",")


def n(valor, arquivo, onde, coluna, **rotulo):
    numeros.append({"n": len(numeros) + 1, "arquivo": arquivo, "linha_ou_chave": onde, "coluna": coluna, **rotulo,
                    "valor": casas(valor)})
    return "%s [%d]" % (casas(valor), len(numeros))


def metricas_de(tabela, arquivo, filtro, rotulo):
    r = tabela[filtro]
    assert len(r) == 1, (arquivo, rotulo)
    i = r.index[0]
    return [n(float(r[m].iloc[0]), arquivo, i + 2, m, modelo=rotulo[0], forma=rotulo[1], ano=rotulo[2]) for m in METRICAS]


md23 = pd.read_csv(PASTA / "sensibilidade_MD_2023.csv")
md24 = pd.read_csv(PASTA / "sensibilidade_MD_2024.csv")
lstm = pd.read_csv(PASTA / "lstm_metricas.csv")
sementes = pd.read_csv(PASTA / "lstm_sementes.csv")
abl23 = pd.read_csv(E / "parte2_ensaio_ablacao_sensibilidade_2023.csv")
met23 = pd.read_csv(E / "parte2_ensaio_metricas.csv")
abl24 = pd.read_csv(E / "parte2_ablacao_sensibilidade_2024.csv")
met24 = pd.read_csv(E / "parte2_metricas.csv")
dobras = pd.read_csv(E / "parte1_sensibilidade_metricas.csv")
assert (pd.concat([md23, md24, lstm])["linhas"] == 1992).all()

# tabela comparativa do item 12: 2023 no corte fixo, 2023 nas 12 dobras e 2024 no corte fixo
linhas = []
for modelo, cf23, d23, cf24 in (
    ("M4 (30 atributos)", metricas_de(abl23, "entradas/parte2_ensaio_ablacao_sensibilidade_2023.csv", abl23["variante"] == "M4", ("M4", "corte fixo", 2023)),
                           metricas_de(dobras, "entradas/parte1_sensibilidade_metricas.csv", dobras["modelo"] == "M4", ("M4", "12 dobras", 2023)),
                           metricas_de(abl24, "entradas/parte2_ablacao_sensibilidade_2024.csv", abl24["variante"] == "M4", ("M4", "corte fixo", 2024))),
    ("MB (28, só RISP)", metricas_de(abl23, "entradas/parte2_ensaio_ablacao_sensibilidade_2023.csv", abl23["variante"] == "MB", ("MB", "corte fixo", 2023)),
                         metricas_de(dobras, "entradas/parte1_sensibilidade_metricas.csv", dobras["modelo"] == "MB", ("MB", "12 dobras", 2023)),
                         metricas_de(abl24, "entradas/parte2_ablacao_sensibilidade_2024.csv", abl24["variante"] == "MB", ("MB", "corte fixo", 2024))),
    ("MD (29, sem bairro)", metricas_de(md23, "sensibilidade_MD_2023.csv", md23["forma"] == "corte_fixo", ("MD", "corte fixo", 2023)),
                            metricas_de(md23, "sensibilidade_MD_2023.csv", md23["forma"] == "dobras_2023", ("MD", "12 dobras", 2023)),
                            metricas_de(md24, "sensibilidade_MD_2024.csv", md24["forma"] == "corte_fixo", ("MD", "corte fixo", 2024))),
    ("LSTM (semente 42)", metricas_de(lstm, "lstm_metricas.csv", lstm["ano"] == 2023, ("LSTM", "corte fixo, conjunto da parada", 2023)),
                          None,
                          metricas_de(lstm, "lstm_metricas.csv", lstm["ano"] == 2024, ("LSTM", "corte fixo", 2024))),
    ("média móvel de 12 meses", metricas_de(met23, "entradas/parte2_ensaio_metricas.csv", (met23["forma"] == "corte_fixo") & (met23["metodo"] == "media_movel_12"), ("média móvel de 12 meses", "corte fixo", 2023)),
                                None,
                                metricas_de(met24, "entradas/parte2_metricas.csv", (met24["forma"] == "corte_fixo") & (met24["metodo"] == "media_movel_12"), ("média móvel de 12 meses", "corte fixo", 2024))),
):
    linhas.append("| %s | %s | %s | %s |" % (modelo, " / ".join(cf23), " / ".join(d23) if d23 else "não se aplica", " / ".join(cf24)))

# sementes da LSTM
sem = []
for i, r in sementes.iterrows():
    sem.append("| %d | %d | %d | %s |" % (r["semente"], r["melhor_epoca"], r["epocas_rodadas"], " / ".join(
        n(float(r[m]), "lstm_sementes.csv", i + 2, m, modelo="LSTM", forma="corte fixo, conjunto da parada", ano=2023, semente=int(r["semente"]))
        for m in METRICAS)))

# tau-b e pesos
rmd = json.loads((PASTA / "resumo_MD.json").read_text(encoding="utf-8"))
rl = json.loads((PASTA / "resumo_lstm.json").read_text(encoding="utf-8"))
taus = [
    "| MD × M4, 27 não espaciais, 12 dobras de 2023 | %s |" % n(rmd["2023"]["tau_b"]["MD x M4"], "resumo_MD.json", "2023/tau_b/MD x M4", "tau_b"),
    "| MD × MB, 27 não espaciais, 12 dobras de 2023 | %s |" % n(rmd["2023"]["tau_b"]["MD x MB"], "resumo_MD.json", "2023/tau_b/MD x MB", "tau_b"),
    "| MD × M4, 27 não espaciais, corte fixo de 2024 | %s |" % n(rmd["2024"]["tau_b"]["MD x M4"], "resumo_MD.json", "2024/tau_b/MD x M4", "tau_b"),
    "| LSTM × M4, 12 defasagens por permutação, 2023 | %s |" % n(rl["tau_b_12_defasagens_lstm_x_m4"], "resumo_lstm.json", "tau_b_12_defasagens_lstm_x_m4", "tau_b"),
]
pesos = ["| %s, %s | %s |" % (m, ano, n(v, "resumo_MD.json", "%s/peso_espacial/%s" % (ano, m), "peso_espacial"))
         for ano in ("2023", "2024") for m, v in rmd[ano]["peso_espacial"].items()]

# permutação
pl = pd.read_csv(PASTA / "permutacao_lstm_2023.csv")
pm = pd.read_csv(PASTA / "permutacao_m4_2023.csv").set_index("atributo")
perm = []
for i, r in pl.iterrows():
    j = pm.index.get_loc(r["defasagem"])
    perm.append("| %s | %d | %s | %d | %s | %d | %d |" % (
        r["defasagem"], r["passo"], n(float(r["aumento_medio"]), "permutacao_lstm_2023.csv", i + 2, "aumento_medio", modelo="LSTM", atributo=r["defasagem"]),
        r["ordem_12"], n(float(pm.loc[r["defasagem"], "aumento_medio"]), "permutacao_m4_2023.csv", j + 2, "aumento_medio", modelo="M4", atributo=r["defasagem"]),
        pm.loc[r["defasagem"], "ordem_12"], pm.loc[r["defasagem"], "ordem_30"]))
outros = ["| %s | %s | %d |" % (a, n(float(pm.loc[a, "aumento_medio"]), "permutacao_m4_2023.csv", pm.index.get_loc(a) + 2, "aumento_medio", modelo="M4", atributo=a), pm.loc[a, "ordem_30"])
          for a in pm.index if a not in set(pl["defasagem"])]

tabela = pd.DataFrame(numeros)
with open(PASTA / "numeros_capitulo.csv", "x", encoding="utf-8", newline="") as f:
    tabela.to_csv(f, index=False)
texto = ["# Números do capítulo: Parte 3 da rodada de 07/10 (bloco 31), MD e LSTM", "",
         "Extraídos por `numeros_capitulo.py` dos arquivos desta pasta (cópias de `rodada_2026-10-07/parte3/`) e de "
         "`entradas/` (cópias da Parte 1 e da Parte 2; sha256 em `HASHES.tsv`). Valores em 6 casas, arredondados da "
         "precisão plena. O número entre colchetes é a linha de `numeros_capitulo.csv`, que traz o arquivo, a linha "
         "(cabeçalho = 1) ou a chave, e a coluna de cada valor. Deviance de Poisson média com piso de 0,01; MAE e RMSE "
         "sem piso; 1.992 linhas de alvo em toda célula.", "",
         "## Tabela comparativa (item 12): MAE / RMSE / deviance", "",
         "Corte fixo: treino com alvos de 2019 a 2022. 12 dobras: janela expansiva mês a mês em 2023. A LSTM em 2023 "
         "é medida no conjunto que decidiu a parada antecipada, não é validação limpa. A média móvel não se ajusta; "
         "o valor de 2023 vale nas 2 formas.", "",
         "| modelo | 2023, corte fixo | 2023, 12 dobras | 2024, corte fixo |", "|---|---|---|---|", *linhas, "",
         "## LSTM: 3 sementes no ensaio (item 8)", "",
         "| semente | melhor época | épocas rodadas | 2023: MAE / RMSE / deviance |", "|---|---|---|---|", *sem, "",
         "## tau-b de Kendall", "", "| comparação | tau-b |", "|---|---|", *taus, "",
         "## Peso espacial (soma das médias |φ| dos atributos espaciais do modelo sobre a soma de todos)", "",
         "| modelo, ano | peso |", "|---|---|", *pesos, "",
         "## Importância por permutação, 2023 (item 10): aumento médio da deviance em 5 permutações", "",
         "| defasagem | passo da LSTM | LSTM | ordem LSTM (12) | M4 | ordem M4 (12) | ordem M4 (30) |",
         "|---|---|---|---|---|---|---|", *perm, "",
         "Demais 18 atributos do M4:", "", "| atributo | aumento | ordem M4 (30) |", "|---|---|---|", *outros]
with open(PASTA / "numeros_capitulo.md", "x", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(texto) + "\n")
print(len(tabela), "números")
