"""Partes comuns das variantes territoriais que completam as 8 combinações de atributos espaciais: ME (bairro e
aisp), MF (só aisp) e MG (nenhum atributo espacial), além das listas da MA (os 27 não espaciais e o bairro) e da MD
(os 27, aisp e risp).

Importa `experimentos/sensibilidade_comum_md.py` (os 27 atributos não espaciais, a MD, o corte fixo, a identificação
das linhas, a média do valor absoluto e o peso espacial) e, por ele, `experimentos/validacao_comum.py` e
`modelo/comum.py`. Os programas que o importam gravam em `resultados/`.

No modo `ensaio`, as linhas de alvo em 2024 saem da base logo depois da leitura. Os programas rodam da raiz do
repositório, por exemplo: python experimentos/sensibilidade_variantes_dobras_2023.py ensaio
"""
import datetime
import hashlib
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import sensibilidade_comum_md as c3  # noqa: E402

c2 = c3.c2
comum = c3.comum
# as saídas de modelo/ e de experimentos/ ficam todas em resultados/
RESULTADOS = c3.RESULTADOS
CONFIG = c3.CONFIG
ALVO = comum.ALVO
ANO = c3.ANO
ESPACIAIS = comum.BLOCOS["espacial"]
NAO_ESPACIAIS = c3.NAO_ESPACIAIS
# as variantes MA (os 27 e o bairro) e MD (os 27, aisp e risp)
MA = comum.em_ordem(NAO_ESPACIAIS + ["bairro"])
MD = c3.MD
VARIANTES = {"ME": comum.em_ordem(NAO_ESPACIAIS + ["bairro", "aisp"]),
             "MF": comum.em_ordem(NAO_ESPACIAIS + ["aisp"]),
             "MG": list(NAO_ESPACIAIS)}
assert len(MA) == 28 and len(MD) == 29
assert [len(v) for v in VARIANTES.values()] == [29, 28, 27]


def agora():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def registrar(texto):
    linha = "%s  %s" % (agora(), texto)
    print(linha, flush=True)


def sha256(caminho):
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def modo():
    """ensaio ou teste, da linha de comando."""
    assert len(sys.argv) == 2 and sys.argv[1] in ANO, "uso: programa.py ensaio|teste"
    m = sys.argv[1]
    return m


def espaciais_de(nomes):
    return [a for a in nomes if a in ESPACIAIS]
