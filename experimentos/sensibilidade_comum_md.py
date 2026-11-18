"""Partes comuns da sensibilidade aos atributos de território: as listas de atributos das variantes MB (os 27 não
espaciais e risp) e MD (os 27, aisp e risp, sem o bairro), o corte fixo, a identificação das linhas, a média do
valor absoluto das contribuições e o peso espacial.

Importa `experimentos/validacao_comum.py` (base, configuração, estimador, métricas, dobras de 2023, gravação) e, por
ele, `modelo/comum.py`. Os programas que o importam gravam em `resultados/`.

No modo `ensaio`, as linhas de alvo em 2024 saem da base logo depois da leitura. Os programas rodam da raiz do
repositório, por exemplo: python experimentos/sensibilidade_md_dobras_2023.py ensaio
"""
import datetime
import hashlib
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import validacao_comum as c2  # noqa: E402

comum = c2.comum
# as saídas de modelo/ e de experimentos/ ficam todas em resultados/
RESULTADOS = c2.RESULTADOS
CONFIG = c2.CONFIG
ALVO = comum.ALVO
ANO = c2.ANO
ANOS_DE_TREINO = c2.ANOS_DE_TREINO
B = comum.BLOCOS
NAO_ESPACIAIS = comum.em_ordem(B["criminal"] + B["temporal"] + B["perfil"])
MB = comum.em_ordem(NAO_ESPACIAIS + ["risp"])
MD = comum.em_ordem(NAO_ESPACIAIS + ["aisp", "risp"])
M4 = comum.atributos()
assert len(NAO_ESPACIAIS) == 27 and len(MB) == 28 and len(MD) == 29 and len(M4) == 30
DEFASAGENS = ["cvli_lag%d" % k for k in range(1, 13)]
CHAVE = ["bairro", "ano_alvo", "mes_alvo"]


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


def ler_base(m):
    return c2.ler_base(m)


def corte_fixo(base, ano):
    """Treino com alvos de 2019 a 2022 (7.968 linhas) e avaliação nas 1.992 linhas do ano."""
    treino = base[base["ano_alvo"].between(*ANOS_DE_TREINO)]
    avaliadas = base[base["ano_alvo"] == ano]
    assert len(treino) == 7968 and len(avaliadas) == 1992
    return treino, avaliadas


def identificacao(df):
    return df[CHAVE].assign(bairro=df["bairro"].astype(str))


def media_abs(contribs, nomes):
    """Média do valor absoluto da contribuição de cada atributo, na escala do log da contagem esperada."""
    return contribs[nomes].abs().mean()


def peso_espacial(medias, espaciais):
    """Soma das médias do valor absoluto dos atributos espaciais do modelo dividida pela soma de todos os dele."""
    return float(medias[[a for a in espaciais if a in medias.index]].sum() / medias.sum())

