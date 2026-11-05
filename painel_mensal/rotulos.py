"""Rótulos, cores e letra que os dois modos do painel usam iguais: a escala dos eixos de contribuição,
as cores do SHAP, os anos da mudança entre anos, a cor e a forma das regiões da dependência e o tamanho
mínimo da letra dos gráficos. Não lê dado nenhum, para o modo normal usar sem importar a demonstração."""

ESCALA = "log da contagem esperada"
EIXO_CONTRIBUICAO = "contribuição (%s)" % ESCALA
# as do waterfall do projeto anterior (SHAP_STYLE_KWARGS em src/utils/shap_chart_data.py): (0,80; 0,25;
# 0,33) aumenta a estimativa e (0,25; 0,45; 0,68) reduz; no enxame, valor alto e valor baixo do atributo
AUMENTA = "#cc4054"
REDUZ = "#4073ad"
# do azul ao vermelho pela mistura dos dois, um roxo, sem cinza no meio
ESCALA_VALOR = [[0, REDUZ], [1, AUMENTA]]
# valor ausente no enxame (janela sem registros ou sem idade informada): cinza-escuro, a 31,6 do cinza dos
# categóricos, 23,2 do azul, 30,3 do vermelho e 20,8 do roxo do meio da escala (OKLab x 100)
COR_AUSENTE = "#33373d"
ANOS = (2019, 2021, 2024)
ROTULO_ANO = {2019: "2019", 2021: "2021 (pandemia)", 2024: "2024"}
# cinzas de claro a escuro, na ordem dos anos, para não lembrar aumento nem redução; menor distância
# OKLab x 100 entre eles de 19,7
CORES_ANO = {2019: "#A6A6A6", 2021: "#6B6B6B", 2024: "#2E2E2E"}
# cor e forma das 3 regiões da dependência, na ordem da legenda. As cores ficam longe do vermelho e do
# azul do SHAP na visão normal (18,0 no mínimo, OKLab x 100); a forma separa o que a cor sozinha não
# separa: na simulação de protanopia a 1ª e a 3ª ficam próximas, e no preto e branco a 2ª e a 3ª têm quase
# a mesma claridade
CORES_REGIAO = ("#E69F00", "#009E73", "#8C8C00")
FORMAS_REGIAO = ("circle", "square", "triangle-up")
# tamanho de cada forma com a área visível do círculo, medida nas imagens do capítulo (densidade 2). No
# plotly o tamanho é o diâmetro do círculo e o lado do quadrado, e o triângulo de mesmo tamanho tem pouco mais
# da metade da área do círculo; como a borda come uma faixa de largura quase fixa, a conta geométrica não
# basta. Pontos: área da cor (o contorno é branco), pela mediana dos marcadores da dependência; o quadrado
# anda em degraus de pixel, e com 7,1 a maioria caía no degrau de baixo. Caso: a silhueta inteira, com o
# contorno preto, em marcadores isolados
TAMANHO_PONTO = dict(zip(FORMAS_REGIAO, (8, 7.3, 11.0)))
TAMANHO_CASO = dict(zip(FORMAS_REGIAO, (12, 10.6, 15.4)))
# no modo normal as regiões são as 3 RISP, na ordem da legenda da figura dos mapas do capítulo
ORDEM_RISP = ("BTS", "CENTRAL", "ATLÂNTICO")
NOMES_RISP = {"BTS": "Baía de Todos os Santos", "CENTRAL": "Central", "ATLÂNTICO": "Atlântico"}
# letra mínima dos gráficos, em px de tela: nas imagens do capítulo a seção sai com cerca de 1046 px (1030
# mais a folga do recorte) e entra com 16 cm, e 14 px dão 6,1 pt
FONTE = 14
# texto dos gráficos (marcas e títulos dos eixos, legendas, barras de cor e anotações), com 7,6:1 contra o
# branco; o cinza do tema do Streamlit, #808495, fica em 3,7:1, abaixo dos 4,5:1 para texto
COR_TEXTO = "#4B5563"


def letra_minima(fig):
    """Marcas e títulos dos eixos, legenda e texto geral do gráfico com FONTE px e COR_TEXTO. O tema do
    Streamlit fixa as marcas dos eixos em 12 px e no cinza dele, e isso passa por cima do texto geral da
    figura."""
    fonte = dict(size=FONTE, color=COR_TEXTO)
    fig.update_xaxes(tickfont=fonte, title_font=fonte)
    fig.update_yaxes(tickfont=fonte, title_font=fonte)
    fig.update_layout(font=fonte, legend_font=fonte)
    return fig
