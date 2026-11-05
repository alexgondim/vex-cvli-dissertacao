"""Item 7 do bloco 26: congela a configuração escolhida e o sha256 de todos os programas da pasta, com data e hora.
Daqui em diante, nenhuma mudança em configuração, atributos ou programas de modelo."""
import json
import sys

import comum

ARQUIVO = comum.PASTA / "CONGELADO.tsv"
if ARQUIVO.exists():
    comum.registrar("PARADA: CONGELADO.tsv já existe e não é regravado")
    sys.exit(1)
escolha = json.loads(comum.ESCOLHA.read_text(encoding="utf-8"))
quando = comum.agora()
linhas = ["item\tnome\tvalor\tdata_hora",
          "configuracao\tordem %d da grade\t%s\t%s" % (escolha["ordem"], json.dumps(escolha["configuracao"]), quando),
          "atributos\t30, na ordem do dicionário\t%s\t%s" % (", ".join(comum.atributos()), quando),
          "base\tpainel_bairro_mes_features.csv\t%s\t%s" % (comum.SHA_BASE, quando)]
for p in sorted(comum.PASTA.glob("*.py")):
    linhas.append("programa\t%s\t%s\t%s" % (p.name, comum.sha256(p), quando))
ARQUIVO.write_text("\n".join(linhas) + "\n", encoding="utf-8")
comum.registrar("etapa 6, congelamento: CONGELADO.tsv com a configuração %d e %d programas; sha256 %s"
                % (escolha["ordem"], len(linhas) - 4, comum.sha256(ARQUIVO)))
