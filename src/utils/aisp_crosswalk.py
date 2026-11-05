"""
Correspondência bairro -> AISP (Área Integrada de Segurança Pública,
Portaria SSP 435/2020, Anexo I) e agregação de valores SHAP por AISP.

Extraído de analises/agregacao_shap_aisp.py (2026-08-16) para reuso pelo
painel -- os dicts de correção, a normalização de nome e a lógica de
merge/agregação são os mesmos, só isolados em função pura. O script de
linha de comando continua funcionando, agora importando daqui em vez de
duplicar a lógica.

CORRECAO DE CODIFICACAO (2026-08-17): a versao anterior deste modulo
lia o CSV com `encoding="latin-1"` e corrigia 21 nomes de bairro
"corrompidos" via dict de excecao (AISP_MOJIBAKE_FIX). Investigacao dos
bytes brutos mostrou que a premissa estava errada -- o corpo do arquivo
e UTF-8 legitimo (mesma codificacao do BOM no inicio do arquivo), nao
uma mistura de latin-1 com pontos corrompidos. Ler o arquivo inteiro
como UTF-8 resolve as 21 entradas do dict SEM excecao nenhuma (141/141
bairros da matriz SHAP encontram AISP so com normalizacao de forma +
o 1 typo real da base criminal + os 2 bairros com citacao normativa
direta) -- e revela 3 nomes de AISP (Sao Caetano, Valeria, Itapua) que
tambem vinham corrompidos e nunca tinham sido notados, porque so
apareciam em texto de relatorio/console, nunca como rotulo de grafico.
AISP_MOJIBAKE_FIX foi removido; _decode_csv_bytes() abaixo tenta UTF-8
no arquivo inteiro e cai para latin-1 por linha (nao silenciosamente)
só se alguma linha realmente não for UTF-8 valido -- protege contra o
arquivo voltar a ficar misto no futuro, sem depender de caso resolvido
a mao.

CRITÉRIO DE CORRESPONDÊNCIA DE NOMES (declarado, não aproximado) -- ver
docstring original em analises/agregacao_shap_aisp.py para a investigação
caso a caso do typo e das 2 citações normativas restantes.
"""
import io
import re
import unicodedata
import warnings
from pathlib import Path
from typing import Dict, List

import pandas as pd

_BASE = Path(__file__).parent.parent.parent
AISP_CSV = _BASE / "data" / "processed" / "AISP_Salvador_bairros_Portaria435-2020_AnexoI.csv"

# 1 erro de digitacao na base criminal (nao na AISP) -- a grafia "Jadim
# Cruzeiro" reproduz fielmente o Anexo I da Portaria 435/2020 conforme
# publicado no DOE; o CSV traz a forma corrigida "Jardim Cruzeiro".
CRIME_TYPO_FIX = {
    "VILA RUY BARBOSA / JADIM CRUZEIRO": "VILA RUY BARBOSA / JARDIM CRUZEIRO",
}

# 5 bairros ausentes do CSV da AISP sob qualquer grafia -- resolvidos por
# citacao normativa direta (nao por normalizacao de nome).
#
# Os 4 "trechos da Paralela" abaixo pertencem TODOS a AISP 11 -- NAO a
# AISP do bairro que da nome ao trecho. O Anexo II da Portaria SSP
# 435/2020 lista 18 trechos da Paralela na composicao da 82a CIPM,
# unidade que responde pelo eixo viario inteiro e integra a AISP 11,
# independente de qual bairro cada trecho margeia (conferido em
# data/raw/crosswalk_bairro_unidade_anexoII.csv). Em 12 dos 18 trechos
# listados no Anexo II, a AISP do trecho (sempre 11) diverge da AISP
# propria do bairro citado no nome -- ex.: bairro Piata e AISP 12,
# bairro Trobogy e AISP 10, bairro Bairro da Paz e AISP 12, mas os tres
# trechos homonimos na base de crimes sao AISP 11. NAO "corrigir" essas
# quatro entradas para a AISP do bairro correspondente -- a atribuicao
# abaixo e a normativa correta, nao um erro a normalizar.
AISP_CORRESPONDENCIA_NORMATIVA = {
    # Anexo II da Portaria SSP 435/2020 -- composicao da 82a CIPM.
    "PARALELA TRECHO PIATA": 11,
    "PARALELA TRECHO TROBOGY": 11,
    "PARALELA TRECHO BAIRRO DA PAZ": 11,
    "PARALELA TRECHO SABOEIRO": 11,
    # Art. 1o da Portaria SSP 002/2021 (retificacao da 435/2020) -- lista
    # Vista Alegre na composicao da AISP 5.
    "VISTA ALEGRE": 5,
}


def norm(name: str) -> str:
    s = str(name).replace("'", " ").replace("-", " ").replace("/", " / ")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.upper().strip()
    return re.sub(r"\s+", " ", s)


def _decode_csv_bytes(raw_bytes: bytes) -> str:
    """Decodifica como UTF-8. Se o arquivo inteiro nao for UTF-8 valido,
    cai para latin-1 SO na(s) linha(s) que falharem -- e avisa quais,
    para o fallback nunca ser silencioso."""
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        linhas_decodificadas = []
        linhas_fallback = []
        for i, linha_bytes in enumerate(raw_bytes.split(b"\n")):
            try:
                linhas_decodificadas.append(linha_bytes.decode("utf-8"))
            except UnicodeDecodeError:
                linhas_decodificadas.append(linha_bytes.decode("latin-1"))
                linhas_fallback.append(i)
        warnings.warn(
            f"aisp_crosswalk: {len(linhas_fallback)} linha(s) do CSV nao "
            f"eram UTF-8 valido, decodificadas como latin-1 (indices de "
            f"linha: {linhas_fallback})"
        )
        return "\n".join(linhas_decodificadas)


def carregar_aisp_csv() -> pd.DataFrame:
    """CSV tem BOM UTF-8 e o corpo tambem e UTF-8 (confirmado -- ver
    docstring do modulo). Descarta o BOM antes de decodificar, senao o
    cabecalho "AISP" vem com lixo grudado."""
    raw_bytes = AISP_CSV.read_bytes()
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        raw_bytes = raw_bytes[3:]
    texto = _decode_csv_bytes(raw_bytes)
    aisp = pd.read_csv(io.StringIO(texto))
    aisp["bairro_norm"] = aisp["bairro"].apply(norm)
    return aisp


def mapear_bairros_para_aisp(bairros: List[str]) -> Dict[str, int]:
    """Resolve o AISP de cada bairro da lista (normalizacao de forma +
    correcao de typo + citacao normativa direta). Bairros nao resolvidos
    ficam de fora do dict retornado."""
    aisp = carregar_aisp_csv()
    aisp_map = dict(zip(aisp["bairro_norm"], aisp["AISP"]))

    aisp_por_bairro: Dict[str, int] = {}
    for b_orig in bairros:
        b_corr = CRIME_TYPO_FIX.get(norm(b_orig), norm(b_orig))
        if b_corr in aisp_map:
            aisp_por_bairro[b_orig] = int(aisp_map[b_corr])
        elif b_corr in AISP_CORRESPONDENCIA_NORMATIVA:
            aisp_por_bairro[b_orig] = AISP_CORRESPONDENCIA_NORMATIVA[b_corr]
    return aisp_por_bairro


def matriz_shap_por_aisp(
    shap_vals, bairros: List[str], feat_labels: List[str]
) -> pd.DataFrame:
    """Matriz AISP x atributos: media da contribuicao SHAP (com sinal, nao
    modulo) dos bairros de cada AISP, por atributo. `feat_labels` define
    as colunas da matriz retornada -- o chamador decide se quer nome
    interno ou rotulo em portugues."""
    aisp_por_bairro = mapear_bairros_para_aisp(bairros)
    bairro_to_idx = {b: i for i, b in enumerate(bairros)}
    registros = []
    for b, a in aisp_por_bairro.items():
        i = bairro_to_idx[b]
        registros.append([a] + list(shap_vals[i]))
    df = pd.DataFrame(registros, columns=["AISP"] + list(feat_labels))
    return df.groupby("AISP")[list(feat_labels)].mean()


def contagem_bairros_por_aisp(bairros: List[str]) -> pd.Series:
    """N de bairros (do conjunto passado) que caem em cada AISP."""
    aisp_por_bairro = mapear_bairros_para_aisp(bairros)
    return pd.Series(list(aisp_por_bairro.values())).value_counts().sort_index()


def nomes_aisp() -> Dict[int, str]:
    """AISP -> nome_aisp (ex.: 16 -> 'Pituba'), direto do CSV."""
    aisp = carregar_aisp_csv()
    return {int(a): nome for a, nome in zip(aisp["AISP"], aisp["nome_aisp"])}
