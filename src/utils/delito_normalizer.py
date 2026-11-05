"""
Normalizacao de texto da coluna DELITO, aplicada antes de qualquer
classificacao (CVLI, latrocinio, feminicidio etc).

Motivo (H-12, Estagio 2.1): o filtro antigo buscava o radical 'LATROCIN'
sem normalizar acento, que nao bate com 'LATROCÍNIO' (33 registros
perdidos na base 2023-2024 processada). Um segundo bug independente:
a extracao SSP-BA de 2023-2025 grava 'ROUBO COM RESULTADO MORTE -
LATROCÍNIO' (sem parenteses); extracoes mais antigas e o processado
atual usam a forma com parenteses '- (LATROCÍNIO)'. Normalizar acento,
caixa e parenteses antes de comparar resolve os dois bugs de uma vez,
em vez de remendar caso a caso com regex.
"""
import re
from unidecode import unidecode


def normalize_delito(raw) -> str:
    """Uppercase, sem acento, sem parenteses, espacos colapsados.

    Ex.: 'ROUBO COM RESULTADO MORTE - (LATROCÍNIO)' e
         'roubo com resultado morte - LATROCÍNIO'
    ambos viram 'ROUBO COM RESULTADO MORTE - LATROCINIO'.
    """
    if raw is None:
        return ""
    try:
        import pandas as pd
        if pd.isna(raw):
            return ""
    except ImportError:
        pass

    text = unidecode(str(raw)).upper().strip()
    text = text.replace("(", "").replace(")", "")
    text = re.sub(r"\s+", " ", text)
    return text
