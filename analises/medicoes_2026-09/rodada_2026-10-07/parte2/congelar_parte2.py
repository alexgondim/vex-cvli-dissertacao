"""Congelamento da Parte 2, depois do ensaio: grava CONGELADO_parte2.tsv com o sha256 do protocolo_parte2.md e de
todos os programas da Parte 2, com data e hora. Só roda com o ensaio conferido e sem congelamento anterior."""
import json

import pandas as pd

import comum2 as c

assert c.PROTOCOLO.exists() and not c.CONGELADO.exists()
assert json.loads((c.PASTA_ENSAIO / "conferencia_ensaio.json").read_text(encoding="utf-8"))["ok"]
quando = c.agora()
arquivos = [c.PROTOCOLO.name] + sorted(p.name for p in c.PARTE2.glob("*.py"))
tabela = pd.DataFrame({"arquivo": arquivos, "sha256": [c.sha256(c.PARTE2 / a) for a in arquivos],
                       "data_hora": quando})
with open(c.CONGELADO, "x", encoding="utf-8", newline="") as f:
    tabela.to_csv(f, sep="\t", index=False)
c.registrar("congelamento: CONGELADO_parte2.tsv gravado em %s, com o protocolo e %d programas (sha256 %s)"
            % (quando, len(arquivos) - 1, c.sha256(c.CONGELADO)))
