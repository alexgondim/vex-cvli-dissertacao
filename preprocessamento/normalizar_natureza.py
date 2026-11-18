"""Normalização do texto da coluna DELITO antes de classificar a natureza: caixa alta, sem acento, sem parênteses e
com espaços colapsados.

Assim, grafias que diferem só em acento, caixa ou parênteses (por exemplo, 'ROUBO COM RESULTADO MORTE - LATROCÍNIO' e
'ROUBO COM RESULTADO MORTE - (LATROCÍNIO)') caem na mesma chave. Não lê nem grava arquivo.

É importado pelo `gerar_base_bairro_mes.py`.
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
