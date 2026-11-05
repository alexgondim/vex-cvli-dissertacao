"""Números do capítulo do teste de 2024 (Parte 2 da rodada de 07/10), extraídos de metricas.csv e de
ablacao_sensibilidade_2024.csv desta pasta: todas as linhas, com MAE, RMSE e deviance em 6 casas.

Grava numeros_capitulo.csv (um número por linha, com o arquivo, a linha do arquivo, contando o cabeçalho como a 1, e
a coluna) e numeros_capitulo.md (os mesmos números em 2 quadros). Recusa sobrescrever. Executar da raiz do projeto."""
from pathlib import Path

import pandas as pd

PASTA = Path(__file__).resolve().parent
METRICAS = ("mae", "rmse", "deviance")
NOME_METRICA = {"mae": "MAE", "rmse": "RMSE", "deviance": "deviance de Poisson"}
FORMA = {"corte_fixo": "corte fixo", "janela": "janela expansiva"}
METODO = {"modelo": "modelo", "binomial_negativa": "binomial negativa", "persistencia": "persistência",
          "media_movel_12": "média móvel de 12 meses", "media_sazonal": "média sazonal"}


def casas(v):
    # o csv guarda a precisão plena; o arredondamento é só aqui
    return f"{v:.6f}".replace(".", ",")


def numeros(arquivo, rotulo):
    t = pd.read_csv(PASTA / arquivo)
    linhas = []
    for i, r in t.iterrows():
        for m in METRICAS:
            linhas.append({"arquivo": arquivo, "linha": i + 2, "coluna": m, **rotulo(r),
                           "metrica": NOME_METRICA[m], "valor": casas(r[m])})
    return t, linhas


metricas, n1 = numeros("metricas.csv", lambda r: {"forma": FORMA[r["forma"]], "ano": int(r["ano"]),
                                                   "metodo_ou_variante": METODO[r["metodo"]]})
ablacao, n2 = numeros("ablacao_sensibilidade_2024.csv",
                      lambda r: {"forma": FORMA[r["forma"]], "ano": int(r["ano"]),
                                 "metodo_ou_variante": "%s (%d atributos)" % (r["variante"], r["atributos"])})
assert len(metricas) == 20 and len(ablacao) == 7
assert (metricas["linhas"] == 1992).all() and (ablacao["linhas"] == 1992).all()
tabela = pd.DataFrame(n1 + n2)
tabela.insert(0, "n", range(1, len(tabela) + 1))
with open(PASTA / "numeros_capitulo.csv", "x", encoding="utf-8", newline="") as f:
    tabela.to_csv(f, index=False)

indice = {(r.arquivo, r.linha, r.coluna): r.n for r in tabela.itertuples()}


def celula(arquivo, linha, m):
    return "%s [%d]" % (casas(m[1]), indice[(arquivo, linha, m[0])])


texto = ["# Números do capítulo: teste de 2024, Parte 2 da rodada de 07/10", "",
         "Extraídos por `numeros_capitulo.py` de `metricas.csv` e `ablacao_sensibilidade_2024.csv` desta pasta, "
         "que são cópias das saídas de `rodada_2026-10-07/parte2/` (sha256 em `HASHES.tsv`). Valores em 6 casas, "
         "arredondados da precisão plena dos arquivos. O número entre colchetes é a linha de `numeros_capitulo.csv`, "
         "que traz o arquivo, a linha do arquivo (cabeçalho = 1) e a coluna de cada valor.", "",
         "- Deviance de Poisson média, com piso de 0,01 na previsão; MAE e RMSE sem piso.",
         "- Toda linha dos 2 quadros tem 1.992 linhas de alvo.",
         "- No corte fixo, o modelo é o de 07/10, treinado com alvos de 2019 a 2022, sem reajuste; a binomial negativa "
         "é ajustada nas mesmas 7.968 linhas. Na janela expansiva, um reajuste por mês, de jan/2022 a dez/2024.", "",
         "## Métricas por forma, ano e método (`metricas.csv`)", "",
         "| forma | ano | método | MAE | RMSE | deviance de Poisson | linha do arquivo |",
         "|---|---|---|---|---|---|---|"]
for i, r in metricas.iterrows():
    texto.append("| %s | %d | %s | %s | %s | %s | %d |" % (
        FORMA[r["forma"]], r["ano"], METODO[r["metodo"]],
        *(celula("metricas.csv", i + 2, (m, r[m])) for m in METRICAS), i + 2))
texto += ["", "## Ablação e sensibilidade territorial no corte fixo, 2024 (`ablacao_sensibilidade_2024.csv`)", "",
          "Cada variante treinada nas 7.968 linhas de alvo de 2019 a 2022, com a configuração 3 da grade.", "",
          "| variante | atributos | MAE | RMSE | deviance de Poisson | linha do arquivo |",
          "|---|---|---|---|---|---|"]
for i, r in ablacao.iterrows():
    texto.append("| %s | %d | %s | %s | %s | %d |" % (
        r["variante"], r["atributos"],
        *(celula("ablacao_sensibilidade_2024.csv", i + 2, (m, r[m])) for m in METRICAS), i + 2))
with open(PASTA / "numeros_capitulo.md", "x", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(texto) + "\n")
print(len(tabela), "números")
