"""Partes comuns da Parte 5 da rodada de 07/10 (bloco 42): sensibilidade territorial completa, com as 3 combinações
de atributos espaciais que faltavam, ME (bairro e aisp), MF (só aisp) e MG (nenhum atributo espacial).

Reaproveita o comum3.py da parte3/ (os 27 atributos não espaciais, a MD, o corte fixo, a identificação das linhas,
a média do valor absoluto e o peso espacial), que importa o comum2.py da parte2/ (base de 27/09 conferida pelo
sha256, configuração 3 da grade, gravação sem sobrescrever) e o comum.py da pasta de cima (atributos, estimador,
métricas com piso de 0,01 só na deviance, 12 dobras de 2023 e pred_contribs). Acrescenta só o log, o modo de
execução com os arquivos da parte5/ e as listas de atributos das variantes. Nada é gravado fora da parte5/.

Da raiz do repositório, sem bytecode:
PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 ./venv/Scripts/python.exe analises/medicoes_2026-09/rodada_2026-10-07/parte5/<programa>.py ensaio|teste

No ensaio, as linhas de alvo em 2024 saem da base logo depois da leitura (comum2.ler_base); no teste, o programa só
roda com o CONGELADO_parte5.tsv gravado e com os arquivos listados nele iguais aos congelados.
"""
import datetime
import hashlib
import sys
from pathlib import Path

import pandas as pd

PARTE5 = Path(__file__).resolve().parent
RODADA = PARTE5.parent
sys.path.insert(0, str(RODADA / "parte3"))
import comum3 as c3  # noqa: E402

c2 = c3.c2
comum = c3.comum
LOG = PARTE5 / "rodada_parte5.log"
PROTOCOLO = PARTE5 / "protocolo_parte5.md"
CONGELADO = PARTE5 / "CONGELADO_parte5.tsv"
CONFIG = c3.CONFIG
ALVO = comum.ALVO
ANO = c3.ANO
ESPACIAIS = comum.BLOCOS["espacial"]
NAO_ESPACIAIS = c3.NAO_ESPACIAIS
# as 2 variantes da conferência de reprodução: MA da Parte 1 (os 27 e o bairro) e MD da Parte 3 (os 27, aisp e risp)
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
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(linha + "\n")
    print(linha, flush=True)


def sha256(caminho):
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def modo():
    """ensaio ou teste, da linha de comando, com as mesmas travas das Partes 2 e 3 sobre os arquivos da parte5/."""
    assert len(sys.argv) == 2 and sys.argv[1] in ANO, "uso: programa.py ensaio|teste"
    m = sys.argv[1]
    if m == "ensaio":
        assert PROTOCOLO.exists(), "o protocolo vem antes do ensaio"
        assert not CONGELADO.exists(), "o ensaio é anterior ao congelamento"
    else:
        assert CONGELADO.exists(), "o teste só roda depois do congelamento"
        for _, l in pd.read_csv(CONGELADO, sep="\t").iterrows():
            assert sha256(PARTE5 / l["arquivo"]) == l["sha256"], "%s mudou depois do congelamento" % l["arquivo"]
    return m


def espaciais_de(nomes):
    return [a for a in nomes if a in ESPACIAIS]
