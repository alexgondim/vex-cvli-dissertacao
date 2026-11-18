"""Desenho das 4 visões de explicação: cascata, enxame, dependência e mudança entre anos.

Recebe os valores, as contribuições e o caso de quem chama. A cascata segue o `shap.plots.waterfall`, com o mesmo par
de cores. Não lê nem grava arquivo. O `app.py` o importa.
"""
import io
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter
from shap.plots._style import style_context

import rotulos

# na tela, a cascata fica com a largura que tem no cartão na janela de 1490 px: como é imagem, esticada numa
# janela larga ela cresceria também na altura e passaria do resumo global ao lado
LARGURA_CASCATA_TELA = 467
# rótulo de valor mais perto que isso, em px de tela, do nome do atributo ou de outro texto se lê junto com ele
# ("cvli_ma6−0,03" parece uma subtração)
DISTANCIA_ROTULO_PX = 6
SEM_REGISTROS = "sem registros na janela"
SEM_IDADE = "sem idade informada"
# com o bairro em escala própria, espaço entre os 2 painéis para as marcas do eixo x do de cima, em px
ESPACO_MARCAS = 30
FUNDO_TARJA = "#fff3cd"
BORDA_TARJA = "#d4a017"
TEXTO_TARJA = "#6b4e00"
TEXTO = "#333333"
# o estilo do waterfall do projeto anterior (SHAP_STYLE_KWARGS em src/utils/shap_chart_data.py), copiado
ESTILO_CASCATA = dict(primary_color_positive=(0.80, 0.25, 0.33), primary_color_negative=(0.25, 0.45, 0.68),
                      secondary_color_positive=(0.90, 0.60, 0.63), secondary_color_negative=(0.60, 0.72, 0.86),
                      hlines_color="#dddddd", vlines_color="#999999", text_color="white",
                      tick_labels_color="#666666")
# atributo categórico no enxame: sem valor alto nem baixo, os pontos ficam num cinza neutro
CINZA_CATEGORIA = "#8a9099"


def numero_br(v, casas=3):
    """Número com vírgula decimal e sinal de menos, sem zeros sobrando no fim."""
    s = ("%.*f" % (casas, v)).rstrip("0").rstrip(".")
    return ("0" if s in ("-0", "") else s).replace("-", "−").replace(".", ",")


def rotulo_valor(v, n_vitimas=None):
    """Valor do atributo; categoria fica como é. Ausente vira texto, nunca NaN: com a janela vazia
    (n_vitimas_hist12 = 0), SEM_REGISTROS; com vítimas na janela, só a idade fica ausente, e é SEM_IDADE."""
    if isinstance(v, str):
        return v
    if pd.isna(v):
        return SEM_IDADE if n_vitimas is not None and n_vitimas > 0 else SEM_REGISTROS
    return numero_br(v)


def rotulo_contribuicao(v):
    """Contribuição com sinal na cascata, como o shap escreve (2 casas, sem os zeros do fim). Se com 2 casas
    ela vira zero, leva as casas, até 4, que mostrem um algarismo diferente de zero; se nem com 4, "0"."""
    for casas in (2, 3, 4):
        s = "%+.*f" % (casas, v)
        if any(c in "123456789" for c in s):
            return re.sub(r"\.?0+$", "", s).replace("-", "−").replace(".", ",")
    return "0"


def _valor_por_linha(valores, max_display):
    """Contribuição de cada linha da cascata, pela posição y em que o shap a desenha: os atributos por ordem de
    |contribuição|, de cima para baixo, e, quando nem todos cabem, a soma dos demais na linha 0."""
    n = min(max_display, len(valores))
    ordem = np.argsort(-np.abs(valores))
    individuais = n if n == len(valores) else n - 1
    linhas = {n - 1 - i: valores[ordem[i]] for i in range(individuais)}
    if individuais < n:
        linhas[0] = valores[ordem[individuais:]].sum()
    return linhas


def _distancia(a, b):
    """Menor distância entre duas caixas, zero quando elas se tocam."""
    return float(np.hypot(max(b.x0 - a.x1, a.x0 - b.x1, 0), max(b.y0 - a.y1, a.y0 - b.y1, 0)))


def _caixas_de_texto(fig, r, exceto):
    """Caixa de cada texto da cascata menos `exceto`: nomes dos atributos, rótulos de valor, estimativa, valor de
    base, marcas do eixo x que ficam dentro dele, escala, tarja e legenda das cores."""
    ax = fig.axes[0]
    x0, x1 = sorted(ax.get_xlim())
    marcas = [t.label1 for t in ax.xaxis.get_major_ticks() if x0 <= t.get_loc() <= x1]
    textos = [t for t in [*ax.get_yticklabels(), *marcas, *ax.texts, *fig.texts] if t is not exceto and t.get_text().strip()]
    return [t.get_window_extent(r) for t in textos] + [a.get_window_extent(r) for a in fig.artists
                                                       if isinstance(a, AnchoredOffsetbox)]


def _afasta_rotulos(fig, ax, rotulos_valor):
    """Rótulo de valor fora da seta a menos de DISTANCIA_ROTULO_PX px de tela de outro texto passa para o outro
    lado da seta, à mesma distância dela; se lá ficar ainda mais perto, ou fora da figura, volta."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    minimo = DISTANCIA_ROTULO_PX * fig.get_figwidth() * fig.dpi / LARGURA_CASCATA_TELA
    setas = [a.get_window_extent(r) for a in ax.patches if type(a).__name__ == "FancyArrow"]
    inv = ax.transData.inverted()

    def folga(txt):
        caixa = txt.get_window_extent(r)
        if caixa.x0 < 0 or caixa.x1 > fig.bbox.width:
            return -1.0
        return min(_distancia(caixa, c) for c in _caixas_de_texto(fig, r, txt))

    for txt in rotulos_valor:
        if txt.get_ha() == "center":
            continue
        antes = folga(txt)
        if antes >= minimo:
            continue
        caixa = txt.get_window_extent(r)
        meio = (caixa.y0 + caixa.y1) / 2
        seta = next((s for s in setas if s.y0 <= meio <= s.y1), None)
        if seta is None:
            continue
        posicao, lado = txt.get_position(), txt.get_ha()
        if lado == "right":
            x, novo = seta.x1 + (seta.x0 - caixa.x1), "left"
        else:
            x, novo = seta.x0 - (caixa.x0 - seta.x1), "right"
        txt.set_position((inv.transform((x, 0))[0], posicao[1]))
        txt.set_ha(novo)
        if folga(txt) < antes:
            txt.set_position(posicao)
            txt.set_ha(lado)


def _ajusta_rotulos(fig, ax):
    """Rótulo de valor que não cabe dentro da seta vai para fora dela, na cor da seta, do lado de costume
    (direita no positivo, esquerda no negativo) ou do outro, se ali não houver espaço: a mesma correção do
    waterfall do projeto anterior, feita depois do tamanho final da figura."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    setas = [a for a in ax.patches if type(a).__name__ == "FancyArrow"]
    inv = ax.transData.inverted()
    eixo = ax.get_window_extent(r)
    margem = 4 / 72 * fig.dpi
    for txt in list(ax.texts):
        if txt.get_color() != "white" or txt.get_ha() != "center":
            continue
        caixa = txt.get_window_extent(r)
        meio = caixa.y0 + caixa.height / 2
        seta = next((a for a in setas if a.get_window_extent(r).y0 <= meio <= a.get_window_extent(r).y1), None)
        if seta is None or caixa.width <= seta.get_window_extent(r).width - 2 * margem:
            continue
        cs = seta.get_window_extent(r)
        positiva = tuple(seta.get_facecolor()[:3]) == ESTILO_CASCATA["primary_color_positive"]
        sobra_esquerda = cs.x0 - margem - caixa.width - eixo.x0
        sobra_direita = eixo.x1 - (cs.x1 + margem + caixa.width)
        if (positiva or sobra_esquerda < 0) and sobra_direita >= 0:
            txt.set_position((inv.transform((cs.x1 + margem, 0))[0], txt.get_position()[1]))
            txt.set_ha("left")
        else:
            txt.set_position((inv.transform((cs.x0 - margem, 0))[0], txt.get_position()[1]))
            txt.set_ha("right")
        txt.set_color(seta.get_facecolor())


def _prende_na_figura(fig, textos, margem=0.02):
    """Desloca para dentro da figura o texto que passaria da borda."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for txt in textos:
        caixa = txt.get_window_extent(r).transformed(fig.transFigure.inverted())
        desvio = max(0, margem - caixa.x0) - max(0, caixa.x1 - (1 - margem))
        if not desvio:
            continue
        if hasattr(txt, "xyann"):
            x, y = txt.xyann
            txt.xyann = (x + desvio * fig.get_figwidth() * 72, y)
        else:
            txt.set_x(txt.get_position()[0] + desvio)


def contagem_esperada(valores, base):
    """Registros esperados do caso: exp do valor de base mais a soma das contribuições, que estão no log da contagem
    esperada. É o número da anotação da cascata e o da linha do mês estimado, no cartão dela."""
    return float(np.exp(base + np.asarray(valores, dtype=float).sum()))


def imagem_cascata(valores, base, linha, nomes, tamanho, max_display=None, tarja=None, dpi=200, n_vitimas=None):
    """PNG da cascata do caso, no estilo do shap.plots.waterfall: "valor = atributo" em cada linha,
    valor de base e estimativa com os nomes usados no painel, a estimativa também em registros esperados, a
    legenda das cores e, se houver, a tarja. Com max_display menor que o número de atributos, os de menor
    contribuição em valor absoluto somam numa barra só, "demais N atributos". `n_vitimas` é o
    n_vitimas_hist12 do caso, que decide o texto do valor ausente."""
    largura, altura = tamanho
    valores = np.asarray(valores, dtype=float)
    exibidos = np.array([rotulo_valor(v, n_vitimas) for v in linha], dtype=object)
    numericos = np.array([np.nan if isinstance(v, str) else v for v in linha], dtype=float)
    explicacao = shap.Explanation(values=valores, base_values=base, data=numericos, display_data=exibidos,
                                  feature_names=list(nomes))
    plt.close("all")
    with style_context(**ESTILO_CASCATA):
        shap.plots.waterfall(explicacao, max_display=max_display or len(nomes), show=False)
    fig = plt.gcf()
    ax = fig.axes[0]
    for eixo in fig.axes[1:]:
        eixo.set_xticks([])  # saem o E[f(X)] e o f(x) do shap; entram os nomes usados no painel
    rotulos_y = ax.get_yticklabels()
    if any(re.fullmatch(r"\d+ other features", t.get_text()) for t in rotulos_y):
        ax.set_yticks(ax.get_yticks(), [re.sub(r"^(\d+) other features$", r"demais \1 atributos", t.get_text())
                                        for t in rotulos_y])
    fig.set_size_inches(largura, altura)
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0 - 0.12 * (x1 - x0), x1 + 0.12 * (x1 - x0))  # espaço para os rótulos de fora da seta
    fx = base + valores.sum()
    anota = dict(xycoords=("data", "axes fraction"), textcoords="offset points", fontsize=12, color=TEXTO,
                 annotation_clip=False)
    estimativa = ax.annotate("estimativa = %s (%s registros esperados)"
                             % (numero_br(fx), numero_br(contagem_esperada(valores, base), 1)),
                             xy=(fx, 1), xytext=(0, 5), ha="right" if fx > sum(ax.get_xlim()) / 2 else "left",
                             va="bottom", **anota)
    valor_de_base = ax.annotate("valor de base = %s" % numero_br(base), xy=(base, 0), xytext=(0, -22),
                                ha="center", va="top", **anota)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: numero_br(v, 2)))
    for txt in ax.texts:
        txt.set_text(txt.get_text().replace(".", ","))
    # os rótulos de valor são os textos simples do eixo; estimativa e valor de base são anotações
    rotulos_valor = [t for t in ax.texts if type(t) is Text]
    por_linha = _valor_por_linha(valores, max_display or len(nomes))
    for txt in rotulos_valor:
        txt.set_text(rotulo_contribuicao(por_linha[round(txt.get_position()[1])]))

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    coluna_de_rotulos = max(rt.get_window_extent(r).width for rt in ax.get_yticklabels()) / fig.dpi
    topo = 0.95 if tarja else 0.55
    fig.subplots_adjust(left=(coluna_de_rotulos + 0.18) / largura, right=0.95, top=1 - topo / altura,
                        bottom=1.3 / altura)
    pos = ax.get_position()
    titulo = fig.text(pos.x0 + pos.width / 2, pos.y0 - 0.62 / altura, rotulos.ESCALA, ha="center", va="top",
                      fontsize=12, color=TEXTO)
    partes = [TextArea(s, textprops=dict(color=c, fontsize=12)) for s, c in
              (("Vermelho", rotulos.AUMENTA), (" aumenta a estimativa; ", TEXTO), ("azul", rotulos.REDUZ),
               (" reduz.", TEXTO))]
    fig.add_artist(AnchoredOffsetbox(loc="lower center", child=HPacker(children=partes, align="baseline", pad=0, sep=0),
                                     bbox_to_anchor=(0.5, 0.012), bbox_transform=fig.transFigure, frameon=False))
    if tarja:
        fig.add_artist(Rectangle((0.02, 1 - 0.44 / altura), 0.96, 0.36 / altura, transform=fig.transFigure,
                                 facecolor=FUNDO_TARJA, edgecolor=BORDA_TARJA, linewidth=0.8))
        fig.text(0.5, 1 - 0.26 / altura, tarja, ha="center", va="center", fontsize=12, color=TEXTO_TARJA)
    _ajusta_rotulos(fig, ax)
    _prende_na_figura(fig, [estimativa, valor_de_base, titulo])
    _afasta_rotulos(fig, ax, rotulos_valor)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def layout_visao(fig, altura, margem_baixo, tarja=None):
    """Altura, margens e letra da visão; com tarja, ela fica no alto, acima da área do gráfico."""
    if tarja:
        fig.add_shape(type="rect", xref="paper", x0=0, x1=1, yref="paper", ysizemode="pixel", yanchor=1, y0=6, y1=34,
                      fillcolor=FUNDO_TARJA, line=dict(color=BORDA_TARJA, width=1), layer="above")
        fig.add_annotation(text=tarja, xref="paper", x=0.5, yref="paper", y=1, yanchor="middle", yshift=20,
                           showarrow=False, font=dict(size=rotulos.FONTE, color=TEXTO_TARJA))
    fig.update_layout(height=altura, margin=dict(l=0, r=0, t=44 if tarja else 10, b=margem_baixo), separators=",.",
                      font=dict(size=rotulos.FONTE))
    return rotulos.letra_minima(fig)


def deslocamentos(valores, meia_largura=0.36, caixas=40):
    """Afastamento vertical de cada ponto no enxame: pontos na mesma faixa de x se empilham para
    cima e para baixo, como no gráfico de enxame do pacote shap, sem sorteio."""
    v = np.asarray(valores, dtype=float)
    faixa = v.max() - v.min()
    caixa = np.round(caixas * (v - v.min()) / faixa) if faixa > 0 else np.zeros(len(v))
    y, camada, anterior = np.zeros(len(v)), 0, None
    for i in np.lexsort((v, caixa)):
        camada = camada + 1 if caixa[i] == anterior else 0
        y[i] = np.ceil(camada / 2) * (1 if camada % 2 else -1)
        anterior = caixa[i]
    maior = np.abs(y).max()
    return y * (meia_largura / maior) if maior > 0 else y


def _faixa_com_zero(valores, folga=0.05):
    """Eixo do menor ao maior valor, sempre com o zero, e folga de 5% da faixa de cada lado."""
    baixo, alto = min(float(np.min(valores)), 0.0), max(float(np.max(valores)), 0.0)
    margem = (alto - baixo) * folga
    return [baixo - margem, alto + margem]


def _dois_paineis(altura, margem_baixo, n):
    """Figura de 2 painéis, um sobre o outro, cada um com o seu eixo x. O de cima tem a altura de 1 dos n atributos
    da figura de painel único, mais ESPACO_MARCAS para as marcas do eixo dele; o de baixo fica com o resto. A
    margem de cima é a de layout_visao sem tarja, 10 px."""
    area = altura - 10 - margem_baixo
    linha = area / n
    return make_subplots(rows=2, cols=1, row_heights=[linha, area - linha - ESPACO_MARCAS],
                         vertical_spacing=ESPACO_MARCAS / area)


def _pontos_enxame(fig, t, phi, nomes, linhas, posicao_do_caso, nome_caso, vitimas, com_barra=True, painel=None):
    """Pontos dos atributos `nomes`, um por linha do eixo y a partir de 0. Com `painel`, os traços vão para essa
    linha da figura de 2 painéis; sem a barra de cores, o traço dos valores só entra se tiver ponto. Devolve as
    contribuições desenhadas."""
    onde = {} if painel is None else dict(row=painel, col=1)
    desenhadas = []
    pontos = {"x": [], "y": [], "cor": [], "texto": []}
    cinza = {"x": [], "y": [], "texto": []}
    ausente = {"x": [], "y": [], "texto": []}
    marca = {"x": [], "y": [], "cor": [], "texto": []}
    marca_cinza = {"x": [], "y": [], "texto": []}
    marca_ausente = {"x": [], "y": [], "texto": []}
    for k, nome in enumerate(nomes):
        contrib = phi.loc[linhas, nome].to_numpy()
        desenhadas += contrib.tolist()
        y = k + deslocamentos(contrib)
        coluna = t.loc[linhas, nome]
        if not pd.api.types.is_numeric_dtype(coluna):
            texto = ["%s = %s" % (nome, v) for v in coluna.astype(str)]
            cinza["x"] += contrib.tolist()
            cinza["y"] += y.tolist()
            cinza["texto"] += texto
            if posicao_do_caso is not None:
                for chave, serie in (("x", contrib), ("y", y), ("texto", texto)):
                    marca_cinza[chave].append(serie[posicao_do_caso])
            continue
        valor = coluna.to_numpy(dtype=float)
        baixo, alto = np.nanpercentile(valor, [5, 95])
        if alto <= baixo:
            baixo, alto = np.nanmin(valor), np.nanmax(valor)
        cor = np.clip((valor - baixo) / (alto - baixo if alto > baixo else 1), 0, 1)
        texto = np.array(["%s = %s" % (nome, rotulo_valor(v, n)) for v, n in zip(valor, vitimas)], dtype=object)
        falta = np.isnan(valor)
        for chave, serie in (("x", contrib), ("y", y), ("cor", cor), ("texto", texto)):
            pontos[chave] += serie[~falta].tolist()
            if chave != "cor":
                ausente[chave] += serie[falta].tolist()
        if posicao_do_caso is not None and falta[posicao_do_caso]:
            for chave, serie in (("x", contrib), ("y", y), ("texto", texto)):
                marca_ausente[chave].append(serie[posicao_do_caso])
        elif posicao_do_caso is not None:
            for chave, serie in (("x", contrib), ("y", y), ("cor", cor), ("texto", texto)):
                marca[chave].append(serie[posicao_do_caso])
    escala = dict(colorbar=dict(title=dict(text="valor do atributo", side="right",
                                           font=dict(size=rotulos.FONTE, color=rotulos.COR_TEXTO)),
                                tickvals=[0, 1], ticktext=["baixo", "alto"],
                                tickfont=dict(size=rotulos.FONTE, color=rotulos.COR_TEXTO),
                                thickness=12, len=0.6)) if com_barra else dict(showscale=False)
    if com_barra or pontos["x"]:
        fig.add_scatter(x=pontos["x"], y=pontos["y"], mode="markers", showlegend=False, text=pontos["texto"],
                        marker=dict(size=7, color=pontos["cor"], cmin=0, cmax=1, colorscale=rotulos.ESCALA_VALOR,
                                    **escala),
                        hovertemplate="%{text}<br>contribuição %{x:.3f}<extra></extra>", **onde)
    if cinza["x"]:
        fig.add_scatter(x=cinza["x"], y=cinza["y"], mode="markers", showlegend=False, text=cinza["texto"],
                        marker=dict(size=7, color=CINZA_CATEGORIA),
                        hovertemplate="%{text}<br>contribuição %{x:.3f}<extra></extra>", **onde)
    if ausente["x"]:
        fig.add_scatter(x=ausente["x"], y=ausente["y"], mode="markers", showlegend=False, text=ausente["texto"],
                        marker=dict(size=7, color=rotulos.COR_AUSENTE),
                        hovertemplate="%{text}<br>contribuição %{x:.3f}<extra></extra>", **onde)
    if marca["x"]:
        fig.add_scatter(x=marca["x"], y=marca["y"], mode="markers", showlegend=False, text=marca["texto"],
                        marker=dict(size=12, color=marca["cor"], cmin=0, cmax=1, colorscale=rotulos.ESCALA_VALOR,
                                    line=dict(color="#000000", width=2)),
                        hovertemplate=nome_caso + "<br>%{text}<br>contribuição %{x:.3f}<extra></extra>", **onde)
    if marca_cinza["x"]:
        fig.add_scatter(x=marca_cinza["x"], y=marca_cinza["y"], mode="markers", showlegend=False,
                        text=marca_cinza["texto"], marker=dict(size=12, color=CINZA_CATEGORIA,
                                                               line=dict(color="#000000", width=2)),
                        hovertemplate=nome_caso + "<br>%{text}<br>contribuição %{x:.3f}<extra></extra>", **onde)
    if marca_ausente["x"]:
        fig.add_scatter(x=marca_ausente["x"], y=marca_ausente["y"], mode="markers", showlegend=False,
                        text=marca_ausente["texto"], marker=dict(size=12, color=rotulos.COR_AUSENTE,
                                                                 line=dict(color="#000000", width=2)),
                        hovertemplate=nome_caso + "<br>%{text}<br>contribuição %{x:.3f}<extra></extra>", **onde)
    return desenhadas


def figura_enxame(t, phi, ordem, linhas, caso, nome_caso, altura, tarja=None, separado=None):
    """Linhas marcadas em `linhas`; cor pelo valor do atributo, entre os percentis 5 e 95 de cada um, do azul
    ao vermelho do SHAP. Atributo categórico fica em cinza, sem escala, e valor ausente em rotulos.COR_AUSENTE.
    O caso (índice de `t`), quando está entre as linhas, aparece em cada linha com ponto maior e contorno preto,
    com a mesma cor.

    Com `separado`, um dos atributos de `ordem`, são 2 painéis, cada um com o seu eixo x: em cima só esse
    atributo, embaixo os demais, na mesma ordem. Cada eixo vai do menor ao maior valor desenhado no painel, com o
    zero e 5% de folga de cada lado; o título do eixo fica no de baixo."""
    indices = list(t.index[linhas])
    posicao_do_caso = indices.index(caso) if caso in indices else None
    vitimas = t.loc[linhas, "n_vitimas_hist12"].tolist() if "n_vitimas_hist12" in t else [None] * len(indices)
    if separado not in ordem:
        fig = go.Figure()
        _pontos_enxame(fig, t, phi, ordem, linhas, posicao_do_caso, nome_caso, vitimas)
        fig.update_yaxes(tickvals=list(range(len(ordem))), ticktext=ordem, autorange="reversed", showgrid=False)
        fig.update_xaxes(title_text=rotulos.EIXO_CONTRIBUICAO, zeroline=True, zerolinecolor="#9aa0a6", showgrid=True,
                         gridcolor="#eceff3")
        return layout_visao(fig, altura, margem_baixo=56, tarja=tarja)
    fig = _dois_paineis(altura, 56, len(ordem))
    for painel, nomes in ((1, [separado]), (2, [a for a in ordem if a != separado])):
        x = _pontos_enxame(fig, t, phi, nomes, linhas, posicao_do_caso, nome_caso, vitimas, com_barra=painel == 2,
                           painel=painel)
        fig.update_yaxes(tickvals=list(range(len(nomes))), ticktext=nomes, range=[len(nomes) - 0.5, -0.5],
                         showgrid=False, row=painel, col=1)
        fig.update_xaxes(range=_faixa_com_zero(x), zeroline=True, zerolinecolor="#9aa0a6", showgrid=True,
                         gridcolor="#eceff3", row=painel, col=1)
    fig.update_xaxes(title_text=rotulos.EIXO_CONTRIBUICAO, row=2, col=1)
    return layout_visao(fig, altura, margem_baixo=56)


def figura_dependencia(t, phi, atributo, linhas, regiao, regioes, caso, nome_caso, altura, titulo_x, tarja=None):
    """Contribuição do atributo contra o valor dele, nas linhas marcadas, com cor e forma pela região.
    `regiao` é a região de cada linha de `t`; `regioes` lista (código, nome na legenda, cor, forma) na ordem da
    legenda. O caso, quando está entre essas linhas, aparece maior e com contorno preto, na cor e na forma da
    região dele."""
    fig = go.Figure()
    for codigo, nome, cor, forma in regioes:
        m = (regiao == codigo) & linhas
        fig.add_scatter(x=t.loc[m, atributo], y=phi.loc[m, atributo], mode="markers", name=nome,
                        marker=dict(size=rotulos.TAMANHO_PONTO[forma], color=cor,
                                    symbol=forma, opacity=0.9, line=dict(width=0.5, color="#ffffff")),
                        hovertemplate=nome + "<br>valor %{x:.3f}<br>contribuição %{y:.3f}<extra></extra>")
    if linhas[caso]:
        _, _, cor, forma = next(r for r in regioes if r[0] == regiao[caso])
        fig.add_scatter(x=[t.loc[caso, atributo]], y=[phi.loc[caso, atributo]], mode="markers", showlegend=False,
                        marker=dict(size=rotulos.TAMANHO_CASO[forma], color=cor,
                                    symbol=forma, line=dict(color="#000000", width=2)),
                        hovertemplate=nome_caso + "<br>valor %{x:.3f}<br>contribuição %{y:.3f}<extra></extra>")
    fig.update_xaxes(title_text=titulo_x, showgrid=True, gridcolor="#eceff3")
    fig.update_yaxes(title_text=rotulos.EIXO_CONTRIBUICAO, zeroline=True, zerolinecolor="#9aa0a6", showgrid=True,
                     gridcolor="#eceff3")
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.17, x=0, font_size=rotulos.FONTE))
    return layout_visao(fig, altura, margem_baixo=90, tarja=tarja)


def figura_anos(ano, phi, ordem, altura, tarja=None, separado=None):
    """Distribuição das contribuições de cada atributo nos anos de rotulos.ANOS, na ordem do resumo global;
    `ano` é o ano do alvo de cada linha de `phi`.

    Com o eixo dos atributos invertido, as caixas de cada atributo ficam de cima para baixo na ordem
    dos anos, a mesma da legenda.

    Com `separado`, são 2 painéis como no enxame: em cima as 3 caixas desse atributo, embaixo as dos demais. Cada
    eixo vai do menor ao maior valor do painel (bigodes e pontos fora das caixas incluídos), com o zero e 5% de
    folga; a legenda fica com os traços do painel de baixo.
    """
    if separado not in ordem:
        fig = go.Figure()
        for a in rotulos.ANOS:
            m = ano == a
            rotulo = rotulos.ROTULO_ANO[a]
            fig.add_box(x=phi.loc[m, ordem].to_numpy().ravel(order="F"), y=np.repeat(ordem, int(m.sum())), name=rotulo,
                        orientation="h", marker=dict(color=rotulos.CORES_ANO[a], size=3), line=dict(width=1.2),
                        hovertemplate="%{y}, " + rotulo + ": %{x:.3f}<extra></extra>")
        fig.update_layout(boxmode="group", boxgroupgap=0.25, boxgap=0.1,
                          legend=dict(orientation="h", yanchor="top", y=-0.17, x=0, font_size=rotulos.FONTE))
        fig.update_yaxes(categoryorder="array", categoryarray=ordem, autorange="reversed")
        fig.update_xaxes(title_text=rotulos.EIXO_CONTRIBUICAO, zeroline=True, zerolinecolor="#9aa0a6", showgrid=True,
                         gridcolor="#eceff3")
        return layout_visao(fig, altura, margem_baixo=90, tarja=tarja)
    fig = _dois_paineis(altura, 90, len(ordem))
    nos_anos = ano.isin(rotulos.ANOS)
    for painel, nomes in ((1, [separado]), (2, [a for a in ordem if a != separado])):
        for a in rotulos.ANOS:
            m = ano == a
            rotulo = rotulos.ROTULO_ANO[a]
            fig.add_box(x=phi.loc[m, nomes].to_numpy().ravel(order="F"), y=np.repeat(nomes, int(m.sum())), name=rotulo,
                        legendgroup=rotulo, showlegend=painel == 2, orientation="h",
                        marker=dict(color=rotulos.CORES_ANO[a], size=3), line=dict(width=1.2),
                        hovertemplate="%{y}, " + rotulo + ": %{x:.3f}<extra></extra>", row=painel, col=1)
        fig.update_yaxes(categoryorder="array", categoryarray=nomes, autorange="reversed", row=painel, col=1)
        fig.update_xaxes(range=_faixa_com_zero(phi.loc[nos_anos, nomes].to_numpy()), zeroline=True,
                         zerolinecolor="#9aa0a6", showgrid=True, gridcolor="#eceff3", row=painel, col=1)
    fig.update_xaxes(title_text=rotulos.EIXO_CONTRIBUICAO, row=2, col=1)
    fig.update_layout(boxmode="group", boxgroupgap=0.25, boxgap=0.1,
                      legend=dict(orientation="h", yanchor="top", y=-0.17, x=0, font_size=rotulos.FONTE))
    return layout_visao(fig, altura, margem_baixo=90)
