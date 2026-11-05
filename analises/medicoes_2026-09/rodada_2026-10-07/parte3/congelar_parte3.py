"""Congelamento da Parte 3, depois do ensaio: grava CONGELADO_parte3.tsv com o sha256 do protocolo, de todos os
programas da parte3/ e de lstm_modelo.pt, com data e hora. Só roda com a conferência da MB ok e sem congelamento."""
import json

import pandas as pd

import comum3 as c

assert c.PROTOCOLO.exists() and not c.CONGELADO.exists()
assert json.loads((c.PARTE3 / "conferencia_MB.json").read_text(encoding="utf-8"))["ok"]
assert (c.PARTE3 / "lstm_modelo.pt").exists() and (c.PARTE3 / "resumo_lstm.json").exists()
quando = c.agora()
arquivos = [c.PROTOCOLO.name] + sorted(p.name for p in c.PARTE3.glob("*.py")) + ["lstm_modelo.pt"]
tabela = pd.DataFrame({"arquivo": arquivos, "sha256": [c.sha256(c.PARTE3 / a) for a in arquivos], "data_hora": quando})
with open(c.CONGELADO, "x", encoding="utf-8", newline="") as f:
    tabela.to_csv(f, sep="\t", index=False)
c.registrar("congelamento: CONGELADO_parte3.tsv gravado em %s, com o protocolo, %d programas e lstm_modelo.pt "
            "(sha256 %s)" % (quando, len(arquivos) - 2, c.sha256(c.CONGELADO)))
