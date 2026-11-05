"""Congelamento da Parte 5, depois do ensaio: grava CONGELADO_parte5.tsv com o sha256 do protocolo, de todos os
programas da parte5/ e das saídas do ensaio que o teste e a tabela leem, com data e hora. Só roda sem congelamento
anterior, com a conferência do item 2 ok e com os ensaios do teste_2024.py e do tabela_combinacoes.py até o fim no
log."""
import json

import pandas as pd

import comum5 as c

SAIDAS_DO_ENSAIO = ["conferencia_2023.json", "sensibilidade_parte5_2023.csv", "contribuicoes_ME_2023.csv",
                    "contribuicoes_MF_2023.csv", "contribuicoes_MG_2023.csv", "importancia_parte5_2023.csv",
                    "resumo_parte5.json"]

assert c.PROTOCOLO.exists() and not c.CONGELADO.exists()
assert json.loads((c.PARTE5 / "conferencia_2023.json").read_text(encoding="utf-8"))["ok"]
log = c.LOG.read_text(encoding="utf-8")
assert "ensaio, item 4: fim" in log and "ensaio, item 6: fim" in log
quando = c.agora()
arquivos = [c.PROTOCOLO.name] + sorted(p.name for p in c.PARTE5.glob("*.py")) + SAIDAS_DO_ENSAIO
tabela = pd.DataFrame({"arquivo": arquivos, "sha256": [c.sha256(c.PARTE5 / a) for a in arquivos], "data_hora": quando})
with open(c.CONGELADO, "x", encoding="utf-8", newline="") as f:
    tabela.to_csv(f, sep="\t", index=False)
c.registrar("congelamento: CONGELADO_parte5.tsv gravado em %s, com o protocolo, %d programas e %d saídas do ensaio "
            "(sha256 %s)" % (quando, len(arquivos) - 1 - len(SAIDAS_DO_ENSAIO), len(SAIDAS_DO_ENSAIO),
                             c.sha256(c.CONGELADO)))
