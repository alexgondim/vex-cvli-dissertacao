"""Números do capítulo da Parte 4 da rodada de 07/10 (bloco 37), ablação com a composição dos blocos de 21/10,
extraídos das cópias desta pasta: a deviance de Poisson de M1 a M4 em 2023 (média das 12 dobras, de
ablacao_blocos2_2023.csv) e em 2024 (corte fixo, de ablacao_blocos2_2024.csv), e o ganho de cada bloco em deviance
(de resumo_parte4.json, conferido contra a diferença calculada aqui a partir dos 2 CSV).

Grava numeros_capitulo.csv (um número por linha, com o arquivo e o local: a linha do CSV, contando o cabeçalho como
a 1, ou a chave do JSON) e numeros_capitulo.md. Recusa sobrescrever. Executar da raiz do projeto."""
import json
from pathlib import Path

import pandas as pd

PASTA = Path(__file__).resolve().parent
MODELOS = ("M1", "M2", "M3", "M4")
BLOCO = {"M2": "temporal", "M3": "espacial", "M4": "perfil"}
ANTERIOR = {"M2": "M1", "M3": "M2", "M4": "M3"}
CHAVE_ANO = {2023: "2023_media_das_12_dobras", 2024: "2024_corte_fixo"}


def casas(v):
    # o csv guarda a precisão plena; o arredondamento é só aqui
    return f"{v:.6f}".replace(".", ",")


v = pd.read_csv(PASTA / "ablacao_blocos2_2023.csv")
t = pd.read_csv(PASTA / "ablacao_blocos2_2024.csv")
resumo = json.loads((PASTA / "resumo_parte4.json").read_text(encoding="utf-8"))
linhas, deviance = [], {}
for ano, tabela, arquivo in ((2023, v, "ablacao_blocos2_2023.csv"), (2024, t, "ablacao_blocos2_2024.csv")):
    for m in MODELOS:
        sel = (tabela["modelo"] == m) & ((tabela["dobra"] == "media_das_12_dobras") if ano == 2023 else True)
        i = tabela.index[sel]
        assert len(i) == 1
        valor = float(tabela.loc[i[0], "deviance"])
        deviance[(ano, m)] = valor
        linhas.append({"arquivo": arquivo, "local": "linha %d" % (i[0] + 2), "coluna": "deviance", "ano": ano,
                       "modelo_ou_bloco": m, "numero": "deviance de Poisson" + (" (média das 12 dobras)" if ano == 2023
                                                                                 else " (corte fixo)"),
                       "valor": casas(valor), "valor_pleno": repr(valor)})
for ano in (2023, 2024):
    for m in ("M2", "M3", "M4"):
        caminho = "ganho_do_bloco/%s/%s/deviance" % (CHAVE_ANO[ano], BLOCO[m])
        valor = resumo["ganho_do_bloco"][CHAVE_ANO[ano]][BLOCO[m]]["deviance"]
        assert abs(valor - (deviance[(ano, m)] - deviance[(ano, ANTERIOR[m])])) < 1e-12, caminho
        linhas.append({"arquivo": "resumo_parte4.json", "local": caminho, "coluna": "deviance", "ano": ano,
                       "modelo_ou_bloco": "%s (%s − %s)" % (BLOCO[m], m, ANTERIOR[m]),
                       "numero": "ganho do bloco em deviance", "valor": casas(valor), "valor_pleno": repr(valor)})
tabela = pd.DataFrame(linhas)
tabela.insert(0, "n", range(1, len(tabela) + 1))
assert len(tabela) == 14
with open(PASTA / "numeros_capitulo.csv", "x", encoding="utf-8", newline="") as f:
    tabela.to_csv(f, index=False)

texto = ["# Números do capítulo: ablação com a composição dos blocos de 21/10 (Parte 4 da rodada de 07/10)", "",
         "Extraídos por `numeros_capitulo.py` de `ablacao_blocos2_2023.csv`, `ablacao_blocos2_2024.csv` e "
         "`resumo_parte4.json` desta pasta, que são cópias das saídas de `rodada_2026-10-07/parte4/` (sha256 em "
         "`HASHES.tsv`). Valores em 6 casas, arredondados da precisão plena dos arquivos. O número entre colchetes é a "
         "linha de `numeros_capitulo.csv`, que traz o arquivo e o local de cada valor.", "",
         "- Deviance de Poisson média, com piso de 0,01 na previsão.",
         "- Blocos de 21/10: criminal (16), temporal (8: os 4 do calendário e as proporções de madrugada, tarde, noite "
         "e fim de semana), espacial (3) e perfil (3: `prop_masc_hist12`, `idade_media_hist12`, `n_vitimas_hist12`). "
         "M1 criminal (16 atributos), M2 + temporal (24), M3 + espacial (27), M4 os 30.",
         "- 2023: média das 12 dobras da janela expansiva (166 linhas por dobra). 2024: corte fixo, treino com alvos "
         "de 2019 a 2022 e avaliação nas 1.992 linhas de 2024. Configuração 3 da grade, semente 42.",
         "- Ganho do bloco: deviance do modelo com o bloco menos a do modelo anterior; negativo é melhora.", "",
         "## Deviance de M1 a M4", "",
         "| modelo | atributos | 2023, média das 12 dobras | 2024, corte fixo |", "|---|---|---|---|"]
ATRIBUTOS = {"M1": 16, "M2": 24, "M3": 27, "M4": 30}
for m in MODELOS:
    a = tabela[(tabela["ano"] == 2023) & (tabela["modelo_ou_bloco"] == m)].iloc[0]
    b = tabela[(tabela["ano"] == 2024) & (tabela["modelo_ou_bloco"] == m)].iloc[0]
    texto.append("| %s | %d | %s [%d] | %s [%d] |" % (m, ATRIBUTOS[m], a["valor"], a["n"], b["valor"], b["n"]))
texto += ["", "## Ganho de cada bloco em deviance", "",
          "| bloco | diferença | 2023, média das 12 dobras | 2024, corte fixo |", "|---|---|---|---|"]
for m in ("M2", "M3", "M4"):
    rot = "%s (%s − %s)" % (BLOCO[m], m, ANTERIOR[m])
    a = tabela[(tabela["ano"] == 2023) & (tabela["modelo_ou_bloco"] == rot)].iloc[0]
    b = tabela[(tabela["ano"] == 2024) & (tabela["modelo_ou_bloco"] == rot)].iloc[0]
    texto.append("| %s | %s − %s | %s [%d] | %s [%d] |" % (BLOCO[m], m, ANTERIOR[m], a["valor"], a["n"], b["valor"],
                                                          b["n"]))
with open(PASTA / "numeros_capitulo.md", "x", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(texto) + "\n")
print(len(tabela), "números")
print("\n".join(texto))
