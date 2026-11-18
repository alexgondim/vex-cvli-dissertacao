"""Saídas do modelo que o painel lê, todas em `resultados/`.

`carregar()` lê `explicacoes.csv` e `explicacoes_info.json`, que o `modelo/exportar_para_painel.py` grava. O primeiro
traz as contribuições nas 11.952 linhas da base por bairro e mês. O segundo traz a configuração, a lista dos atributos
e a escala. Sem `explicacoes.csv`, `carregar()` devolve None, e as visões de explicação ficam reservadas.

`carregar_metricas()` lê `validacao_metricas.csv` e `validacao_metricas_mensais.csv`, as métricas do teste de 2024. O
`experimentos/validacao_juntar_metricas.py` e o `experimentos/validacao_janela_expansiva.py` gravam esses 2 arquivos
no modo `teste`. Se faltar um dos 2, `carregar_metricas()` devolve None.

O módulo não treina modelo, não calcula métrica e não grava arquivo. O `app.py` o importa com o nome `modelo`.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PASTA = Path(__file__).resolve().parents[1] / "resultados"
CATEGORICAS = ("bairro", "aisp", "risp")
ANO_EXPLICADO = 2024  # o enxame, a dependência e o mapa usam as linhas do teste do corte fixo
TOTAL = "total"


@dataclass
class Rodada:
    linhas: pd.DataFrame     # bairro, ano_alvo, mes_alvo, divisao, risp, valor_base, margem, estimativa
    valores: pd.DataFrame    # os 30 atributos, categóricos como texto
    phi: pd.DataFrame        # as 30 contribuições, na escala do log da contagem esperada
    info: dict

    @property
    def atributos(self):
        return list(self.info["atributos"])


def carregar():
    if not (PASTA / "explicacoes.csv").exists():
        return None
    e = pd.read_csv(PASTA / "explicacoes.csv")
    info = json.loads((PASTA / "explicacoes_info.json").read_text(encoding="utf-8"))
    nomes = info["atributos"]
    valores = e[["valor_" + a for a in nomes]].set_axis(nomes, axis=1)
    for a in CATEGORICAS:
        valores[a] = valores[a].astype(str)
    phi = e[["contrib_" + a for a in nomes]].set_axis(nomes, axis=1)
    linhas = e[["bairro", "ano_alvo", "mes_alvo", "divisao", "risp", "valor_base", "margem", "estimativa"]]
    return Rodada(linhas, valores, phi, info)


@dataclass
class Metricas:
    anuais: pd.DataFrame     # forma, ano, metodo, linhas, mae, rmse, deviance (20 linhas)
    mensais: pd.DataFrame    # ano, mes, metodo, linhas, mae, rmse, deviance (36 meses e 5 métodos)


def carregar_metricas():
    arquivos = [PASTA / n for n in ("validacao_metricas.csv", "validacao_metricas_mensais.csv")]
    if not all(a.exists() for a in arquivos):
        return None
    return Metricas(pd.read_csv(arquivos[0]), pd.read_csv(arquivos[1]))


def indice(r, bairro, periodo):
    i = r.linhas.index[(r.linhas["bairro"] == bairro) & (r.linhas["ano_alvo"] == periodo.year)
                       & (r.linhas["mes_alvo"] == periodo.month)]
    assert len(i) == 1, (bairro, periodo)
    return i[0]


def no_ano_explicado(r):
    return r.linhas["ano_alvo"] == ANO_EXPLICADO


def ordem_global(r):
    """Atributos pela média do valor absoluto das contribuições nas linhas de 2024, do maior ao menor."""
    return r.phi[no_ano_explicado(r)].abs().mean().sort_values(ascending=False).index.tolist()


def media_abs(r, atributo, agregacao):
    """Média do valor absoluto da contribuição do atributo, ou do total (soma dos valores absolutos dos 30 em
    cada linha), nas linhas de 2024, por bairro, por AISP ou por RISP. A AISP de cada linha é o valor do atributo
    aisp dela, a AISP do bairro."""
    m = no_ano_explicado(r)
    absoluto = r.phi[m].abs().sum(axis=1) if atributo == TOTAL else r.phi.loc[m, atributo].abs()
    chave = r.valores.loc[m, "aisp"].astype(int) if agregacao == "aisp" else r.linhas.loc[m, agregacao]
    return absoluto.groupby(chave).mean()
