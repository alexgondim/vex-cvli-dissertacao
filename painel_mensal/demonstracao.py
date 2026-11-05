"""Modo de demonstração do painel: as 4 visões de explicação preenchidas por um XGBoost treinado em
dado sintético, só para mostrar a disposição delas antes da execução do modelo. Não é resultado.

O único arquivo lido aqui é o painel sintético de exemplo da proposta do orientador (6 bairros,
432 linhas). Bairros, regiões e atributos ganham nomes fictícios, e este módulo não usa dados.py:
nada da base real entra na demonstração. O modo normal do painel não importa este módulo.

O modelo usa só os 8 atributos numéricos. Bairro e região ficam fora das entradas, para o resumo
global não repetir o padrão da rodada real, em que o bairro fica no topo: a região só colore a
dependência, e o bairro só escolhe a linha da explicação local.

O desenho das 4 visões é o de visoes_shap.py, o mesmo do modo normal, com a tarja e os nomes fictícios.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xgboost as xgb

import rotulos
import visoes_shap

ARQUIVO_SINTETICO = (Path(__file__).resolve().parents[1] / "analises" / "medicoes_2026-09" / "orientador"
                     / "painel_bairro_mes_features.csv")
# 8 colunas numéricas do arquivo, na ordem dele: uma de cada tipo, sem par redundante (as 12
# defasagens, as 3 médias móveis, seno e cosseno, ano e índice de tempo andam juntos)
COLUNAS_DE_ORIGEM = ["month_sin_alvo", "time_index", "cvli_lag1", "cvli_ma12", "cvli_trend",
                     "prop_noturno_hist12", "prop_fds_hist12", "idade_media_hist12"]
ATRIBUTOS = ["atributo %d" % (i + 1) for i in range(len(COLUNAS_DE_ORIGEM))]
ABAS_DEPENDENCIA = ATRIBUTOS[:4]
ULTIMO_ANO_DE_TREINO = 2022
# o enxame e a dependência usam só as linhas deste ano, como o modo normal usa as do teste
ANO_EXPLICADO = 2024
BAIRRO_EXEMPLO = "Bairro fictício 01"
MES_EXEMPLO = pd.Period("2024-06", freq="M")
PARAMETROS = {"objective": "count:poisson", "tree_method": "hist", "random_state": 42}

TITULO_SECAO = "Demonstração com dado sintético"
TARJA = "Dado sintético: mostra a disposição, não é resultado."
TEXTO_ANOS = ("Modelo de demonstração, treinado com alvos sintéticos de 2019 a 2022: 2019 e 2021 dentro do treino, "
              "2024 fora.")
IDENTIFICACAO_LINHAS = "Modelo de demonstração. Linhas de 2024, fora do treino."

REGIOES = ("Região A", "Região B", "Região C")
CORES_REGIAO = dict(zip(REGIOES, rotulos.CORES_REGIAO))
FORMAS_REGIAO = dict(zip(REGIOES, rotulos.FORMAS_REGIAO))
# preenchimento das barras de cada região na agregação: liso, hachurado e pontilhado, para as 3 se separarem
# também no preto e branco
PADROES_REGIAO = dict(zip(REGIOES, ("", "/", ".")))
TITULO_AGREGACAO = "Onde a contribuição pesa"
CARTAO_AGREGACAO = "Agregação por região"
EIXO_AGREGACAO = "média de |contribuição| (%s)" % rotulos.ESCALA
# polegadas da figura da cascata; mostrada na largura do cartão (cerca de 480 px de tela nas imagens do
# capítulo), a letra de 12 pt da figura fica com cerca de 6,8 pt no capítulo
TAMANHO_CASCATA = (5.2, 5.6)


def dados_sinteticos():
    """Painel sintético com nomes fictícios: os nomes de origem de bairro e região não passam daqui.

    As colunas de mês-alvo, alvo e tamanho da janela do perfil mantêm os nomes da base, para a linha
    do caso ser lida pela mesma função do modo normal (dados.linha_da_base).
    """
    d = pd.read_csv(ARQUIVO_SINTETICO)
    bairros = {b: "Bairro fictício %02d" % (i + 1) for i, b in enumerate(sorted(d["bairro"].unique()))}
    regioes = sorted(d["risp"].unique())
    assert len(regioes) == 3, regioes
    regioes = dict(zip(regioes, REGIOES))
    t = pd.DataFrame({"bairro": d["bairro"].map(bairros), "região": d["risp"].map(regioes),
                      "ano_alvo": d["ano_alvo"], "mes_alvo": d["mes_alvo"]})
    for nome, coluna in zip(ATRIBUTOS, COLUNAS_DE_ORIGEM):
        t[nome] = d[coluna]
    t["target_cvli"] = d["target_cvli"]
    t["n_vitimas_hist12"] = d["n_vitimas_hist12"]
    return t


def modelo_e_contribuicoes():
    """Treina com os meses-alvo até ULTIMO_ANO_DE_TREINO, só com os 8 atributos, e devolve as
    contribuições de todas as linhas.

    As contribuições saem de pred_contribs, na escala da margem do count:poisson (o log da contagem
    esperada); a última coluna é o valor de base, igual em todas as linhas. Nenhuma métrica de
    desempenho é calculada.
    """
    t = dados_sinteticos()
    X = t[ATRIBUTOS]
    treino = t["ano_alvo"] <= ULTIMO_ANO_DE_TREINO
    modelo = xgb.XGBRegressor(**PARAMETROS)
    modelo.fit(X[treino], t.loc[treino, "target_cvli"])
    contrib = modelo.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    # soma das contribuições = margem da previsão: confere que a escala é a do log da contagem esperada
    assert np.allclose(contrib.sum(axis=1), modelo.predict(X, output_margin=True), atol=1e-4)
    assert np.allclose(contrib[:, -1], contrib[0, -1])
    phi = pd.DataFrame(contrib[:, :-1], columns=ATRIBUTOS, index=t.index)
    return t, phi, float(contrib[0, -1]), modelo


def indice_do_caso(t):
    """Linha de BAIRRO_EXEMPLO em MES_EXEMPLO: a da explicação local e a marcada no enxame."""
    linha = t.index[(t["bairro"] == BAIRRO_EXEMPLO) & (t["ano_alvo"] == MES_EXEMPLO.year)
                    & (t["mes_alvo"] == MES_EXEMPLO.month)]
    assert len(linha) == 1
    return linha[0]


def identificacao_local(mes_alvo):
    return "Modelo de demonstração. Linha %s do treino." % ("dentro" if mes_alvo.year <= ULTIMO_ANO_DE_TREINO
                                                           else "fora")


def ordem_global(t, phi):
    """Atributos pela média do valor absoluto das contribuições nas linhas de ANO_EXPLICADO, do maior ao menor."""
    return phi[t["ano_alvo"] == ANO_EXPLICADO].abs().mean().sort_values(ascending=False).index.tolist()


def imagem_cascata(t, phi, base, dpi=200):
    """PNG da cascata do caso de demonstração, com os 8 atributos e a tarja."""
    caso = indice_do_caso(t)
    return visoes_shap.imagem_cascata(phi.loc[caso, ATRIBUTOS].to_numpy(dtype=float), base,
                                      t.loc[caso, ATRIBUTOS].astype(float).tolist(), ATRIBUTOS, TAMANHO_CASCATA,
                                      tarja=TARJA, dpi=dpi)


def com_tarja(fig, altura, margem_baixo):
    """Tarja no alto da visão, acima da área do gráfico."""
    return visoes_shap.layout_visao(fig, altura, margem_baixo, tarja=TARJA)


def agregacao_por_regiao(t, phi, ordem):
    """Média do valor absoluto da contribuição de cada atributo nas linhas de ANO_EXPLICADO de cada região,
    na ordem do resumo global. A agregação dispensa o mapa: os bairros fictícios não têm polígono."""
    linhas = t["ano_alvo"] == ANO_EXPLICADO
    return pd.DataFrame({r: phi.loc[linhas & (t["região"] == r), ordem].abs().mean() for r in REGIOES})


def figura_agregacao(t, phi, ordem, altura):
    """Barras agrupadas: para cada atributo, uma barra por região, com a cor e o preenchimento dela."""
    media = agregacao_por_regiao(t, phi, ordem)
    fig = go.Figure()
    for regiao in REGIOES:
        fig.add_bar(x=list(ordem), y=media[regiao], name=regiao,
                    marker=dict(color=CORES_REGIAO[regiao],
                                pattern=dict(shape=PADROES_REGIAO[regiao], fillmode="overlay", fgcolor="#ffffff",
                                             size=8, solidity=0.35)),
                    hovertemplate=regiao + ", %{x}: %{y:.3f}<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.06,
                      legend=dict(orientation="h", yanchor="top", y=-0.14, x=0, font_size=rotulos.FONTE))
    fig.update_yaxes(title_text=EIXO_AGREGACAO, rangemode="tozero", showgrid=True, gridcolor="#eceff3")
    fig.update_xaxes(showgrid=False)
    return com_tarja(fig, altura, margem_baixo=70)


def figura_enxame(t, phi, ordem, altura):
    """Linhas de ANO_EXPLICADO, com o caso de demonstração marcado."""
    return visoes_shap.figura_enxame(t, phi, ordem, t["ano_alvo"] == ANO_EXPLICADO, indice_do_caso(t), BAIRRO_EXEMPLO,
                                     altura, tarja=TARJA)


def figura_dependencia(t, phi, atributo, altura):
    """Linhas de ANO_EXPLICADO, com cor e forma pela região fictícia."""
    regioes = [(r, r, CORES_REGIAO[r], FORMAS_REGIAO[r]) for r in REGIOES]
    return visoes_shap.figura_dependencia(t, phi, atributo, t["ano_alvo"] == ANO_EXPLICADO, t["região"], regioes,
                                          indice_do_caso(t), BAIRRO_EXEMPLO, altura, "valor do %s" % atributo,
                                          tarja=TARJA)


def figura_anos(t, phi, ordem, altura):
    return visoes_shap.figura_anos(t["ano_alvo"], phi, ordem, altura, tarja=TARJA)
