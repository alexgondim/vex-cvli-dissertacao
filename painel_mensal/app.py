"""Mock do painel mensal do VEX-CVLI, para discussão com o orientador.

O painel não treina nada. A barra lateral tem só o caso explicado (bairro e mês-alvo), que aparece só com a aba de
explicação aberta, e a área principal, 3 abas. A exploração (seleção própria de bairros, vazia por padrão, filtros,
mapa no modo observado, distribuições e série mensal filtrada) lê a base de 27/09; as distribuições comparam a
seleção com Salvador, e a série soma os bairros da seleção (sem seleção, Salvador inteira).
Ao lado do mapa ficam o turno e a idade da vítima, e na linha de baixo, a natureza, o gênero da vítima e a RISP. Turno
e natureza ficam em barras, idade em histograma por faixa, gênero da vítima em rosca de 2 anéis, com o bairro por fora
e Salvador por dentro, e RISP em pizza. A explicação (faixa de contexto, as 4 visões e mapa no modo contribuição,
por bairro, AISP ou RISP) lê as saídas do modelo do corte fixo da rodada de 07/10, em dados_modelo/ (modelo.py); sem
essa pasta, fica como espaço reservado. A aba de diagnóstico lê os arquivos do teste de 2024, gravados pela Parte 2
em dados_modelo/.

Da raiz do projeto: ./venv/Scripts/python.exe -m streamlit run painel_mensal/app.py
O .streamlit/config.toml desta pasta fixa o servidor em localhost e sem estatísticas de uso, e
esconde a barra de ferramentas do Streamlit; se localhost e estatísticas desligadas não estiverem
valendo, o mock não carrega dado nenhum.

Com o parâmetro de demonstração (... app.py -- --demonstracao), a aba de explicação mostra só as 4 visões de um
modelo treinado em dado sintético (demonstracao.py), com tarja em cada uma, e a agregação por região; as outras
abas não mudam.
Sem o parâmetro, o mock não importa o módulo de demonstração.
"""
import math
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import dados
import modelo
import rotulos
import visoes_shap

DEMONSTRACAO = "--demonstracao" in sys.argv[1:]
if DEMONSTRACAO:
    import demonstracao

RESERVADO = "Aguarda a execução do modelo."
DIAGNOSTICO = ("métricas do modelo (MAE, RMSE, deviance de Poisson) contra persistência, média sazonal, "
               "média móvel de 12 meses e binomial negativa; rodada: a definir")
ABAS_DEPENDENCIA = ("madrugada", "tarde", "noite", "tendência")
# nome por extenso de cada atributo das abas, como na tabela de atributos do capítulo
EIXO_DEPENDENCIA = {"madrugada": "proporção de registros na madrugada (12 meses)",
                    "tarde": "proporção de registros na tarde (12 meses)",
                    "noite": "proporção de registros na noite (12 meses)",
                    "tendência": "tendência (contagem do mês t menos a do mês t−1)"}
ATRIBUTO_DA_ABA = {"madrugada": "prop_madrugada_hist12", "tarde": "prop_tarde_hist12",
                   "noite": "prop_noite_hist12", "tendência": "cvli_trend"}
# corte fixo: treino com os meses-alvo até 2022, validação em 2023, teste em 2024
ULTIMO_ANO_DE_TREINO = 2022
ANO_DE_TESTE = 2024
# os cartões têm só título; o que é cada visão, cada métrica e cada forma de validação fica no texto do capítulo
CABECALHO = "Proposta de interface. As visões de explicação aguardam a execução do modelo."
CABECALHO_COM_MODELO = ("Proposta de interface. As visões de explicação mostram o modelo do corte fixo; a estimativa na "
                        "série e o diagnóstico ficam para depois do teste de 2024.")
ABAS = ("Exploração", "Explicação", "Diagnóstico")
# a série da aba de explicação cobre o período todo, sem os filtros da exploração
PERIODO_TODO = (dados.MESES[0], dados.MESES[-1])
# no seletor do mapa no modo contribuição, a soma dos 30 (modelo.TOTAL)
ROTULO_TOTAL = "todos os atributos"
# atributos no enxame e na mudança entre anos, com a chave fechada; a cascata mostra 1 a menos e soma os demais
MOSTRADOS = 10
# a cascata tem a largura da da demonstração (as capturas medem a letra por ela) e cresce com o número de barras
LARGURA_CASCATA = 5.2
TEXTO_CINZAS = ("Cinza: atributo categórico, sem valor alto ou baixo. Cinza-escuro: valor ausente, por janela sem "
                "registros ou sem idade informada.")
# entrada da estimativa na legenda da série: sem o modelo, e na demonstração, a de antes da rodada
LEGENDA_ESTIMATIVA = "estimativa: após a execução do modelo"
LEGENDA_ESTIMATIVA_COM_MODELO = "estimativa: entra depois do teste de 2024"
# com os arquivos do teste de 2024 em dados_modelo/: estimativa na série e aba de diagnóstico, sem cabeçalho
LEGENDA_ESTIMATIVA_DESENHADA = "estimativa (modelo do corte fixo)"
COR_ESTIMATIVA = "#4b5563"
ANO_DO_TESTE = 2024
METODOS = ("modelo", "binomial_negativa", "persistencia", "media_movel_12", "media_sazonal")
REFERENCIAS = {"binomial_negativa": "binomial negativa", "persistencia": "persistência",
               "media_movel_12": "média móvel de 12 meses", "media_sazonal": "média sazonal"}
NOMES_CORTE_FIXO = {"modelo": "modelo do corte fixo", **REFERENCIAS}
NOMES_JANELA = {"modelo": "modelo, reajustado a cada mês", **REFERENCIAS}
NOMES_METRICAS = {"mae": "MAE", "rmse": "RMSE", "deviance": "deviance de Poisson"}  # tabela e controle
NOMES_LONGOS = {"mae": "erro absoluto médio (MAE)", "rmse": "raiz do erro quadrático médio (RMSE)",
                "deviance": "deviance de Poisson"}  # títulos dos painéis e eixo dos gráficos
CINZA_BARRA = "#9aa0a6"
CINZA_LINHA = "#6b7280"
# traço e marcador de cada referência nas linhas da janela; o modelo é a linha azul contínua
ESTILO_REFERENCIA = {"binomial_negativa": ("dash", "square"), "persistencia": ("dot", "diamond"),
                     "media_movel_12": ("dashdot", "triangle-up"), "media_sazonal": ("longdash", "x")}
COR_TESTE = "#f6c6c0"
ALTURA_BARRAS = 300
ALTURA_JANELA = 400
ALTURA_MENSAL = 440
REGIOES_RISP = [(r, rotulos.NOMES_RISP[r], c, f) for r, c, f in zip(rotulos.ORDEM_RISP, rotulos.CORES_REGIAO,
                                                                     rotulos.FORMAS_REGIAO)]
# os 2 mapas, o da exploração e o da contribuição, com a AISP no meio
ROTULO_AGREGACAO = {"bairro": "bairro", "aisp": "AISP", "risp": "RISP"}
FAIXAS_SERIE = (
    (pd.Period("2020-01", freq="M"), pd.Period("2021-12", freq="M"), "pandemia", "#fbe0a0"),
    (pd.Period("2024-01", freq="M"), pd.Period("2024-12", freq="M"), "teste", "#f6c6c0"),
)

AZUL = "#1f4e79"
CINZA_CIDADE = "#c3c9d2"
# fatias da rosca do gênero da vítima, iguais nos 2 anéis e sem azul, que no painel marca o bairro e a RISP dele
COR_MASCULINO = "#d1d5db"
COR_FEMININO = "#c2410c"
CINZA_POLIGONO = "#b9bdc4"
CINZA_CONTORNO = "#374151"
VERMELHO = "#c0392b"
# bairro selecionado em preto, porque as legendas das figuras do capítulo falam em contorno preto
CONTORNO_SELECIONADO = "#000000"
LARGURA_CONTORNO_SELECIONADO = 3
FUNDO_RESERVADO = "#f1f3f5"
# ColorBrewer Reds sem o primeiro degrau, quase branco, que some no fundo cinza-claro
ESCALA_VERMELHA = ["#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d", "#a50f15", "#67000d"]
# modo contribuição: ColorBrewer Oranges, também sem o primeiro degrau
ESCALA_LARANJA = ["#fee6ce", "#fdd0a2", "#fdae6b", "#fd8d3c", "#f16913", "#d94801", "#a63603", "#7f2704"]
# o carto-positron dos traços mapbox usa blocos raster da CARTO, que passaram a exigir chave (saem com a
# marca "API KEY REQUIRED"); o dos traços map (MapLibre) usa os blocos vetoriais, que não exigem. Sem nomes:
# com o quadro nos bairros, o "SALVADOR" do mapa de fundo caía em cima deles
ESTILO_MAPA = "carto-positron-nolabels"
TRANSPARENTE = [[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]]
ALTURA_MAPA = 560
# o quadro do mapa se ajusta aos 166 bairros (dados.enquadramento). As ilhas dos Frades e de Bom Jesus dos
# Passos, sem registro, ficam cerca de 8 km a oeste do bairro mais a oeste e saem do quadro do cartão; quem
# quiser vê-las arrasta o mapa. As legendas ficam onde não há bairro. A barra de cor fica à esquerda, sobre a
# baía, entre a Ilha de Maré e a península de Itapagipe, com o topo TOPO_LEGENDA px abaixo da borda: no mapa
# de 570 px da janela de 1490, é a altura que a deixa mais longe dos polígonos (16,5 px). A dos polígonos
# sem cor fica no canto de baixo à direita, sobre o mar, ATRIBUICAO px acima da borda, logo acima da
# atribuição do mapa de fundo (34 px com a margem; 40 px nas figuras do mapa do capítulo, com a letra maior)
TOPO_LEGENDA = 134
ATRIBUICAO = 46
# série na largura inteira, com a área do gráfico na proporção da agregação por região da demonstração, 2,8
# para 1 (947 × 340 px na janela de 1490)
ALTURA_SERIE = 437
# ao lado do mapa, o turno em cima e a idade da vítima embaixo, em 2 cartões com gráficos desta altura: com o título,
# a borda e o espaçamento padrão entre os cartões, a coluna fecha na altura do cartão do mapa no modo observado (668 px
# na janela de 1490)
ALTURA_AO_LADO_DO_MAPA = 248
# histograma da idade: em cada faixa, a barra de Salvador atrás, com 85% da largura da faixa, e a do bairro na frente,
# com 45%, as 2 centradas
LARGURA_IDADE_CIDADE = 0.85
LARGURA_IDADE_BAIRRO = 0.45
# distribuições na linha abaixo do mapa, do turno e da idade, todas com esta altura: natureza em barras horizontais,
# porque são categorias sem ordem e com nomes longos (o turno e a idade ficam verticais, porque têm ordem), gênero da
# vítima em rosca de 2 anéis, com o bairro por fora e Salvador por dentro, e RISP em pizza
ALTURA_DISTRIBUICAO = 348
# margem de cima da natureza, da RISP e da idade: 3 linhas de legenda com a letra de 14 px, porque a entrada do bairro
# de nome longo ocupa 2 (NOME_LONGO); com ela, a área das barras da natureza fica com 216 px
TOPO_DISTRIBUICAO = 74
# espessura das barras da natureza: 4 categorias de 2 barras, com o bargap (0,25) e o bargroupgap (0,06) do turno
CATEGORIAS_DA_NATUREZA = 4
LARGURA_BARRA = (1 - 0.25) / 2 * (1 - 0.06)
CENTRO_BAIRRO = -(1 - 0.25) / 4
# nomes das categorias, no eixo da natureza, na legenda da rosca do gênero da vítima e nas fatias da RISP; os longos
# em 2 linhas
EIXO_NATUREZA = {"Homicídio Doloso": "homicídio doloso", "Feminicídio": "feminicídio", "Latrocínio": "latrocínio",
                 "Lesão Corporal Seguida de Morte": "lesão corporal<br>seguida de morte"}
EIXO_SEXO = {"M": "masculino", "F": "feminino"}
EIXO_RISP = {"BTS": "Baía de Todos<br>os Santos", "CENTRAL": "Central", "ATLÂNTICO": "Atlântico"}
# com nome de bairro maior que isto, a entrada dele na legenda das distribuições quebra em 2 linhas: sem a quebra,
# a de Granjas Rurais Presidente Vargas e a de Vila Ruy Barbosa/Jardim Cruzeiro passavam da largura no gênero da
# vítima e na RISP
NOME_LONGO = 24
# pizza da RISP e rosca do gênero da vítima: o texto vai dentro da fatia quando cabe na horizontal (cabe_dentro, a conta
# do plotly); na pizza, só a partir de FATIA_DENTRO. O que não cabe vai para fora, com linha de chamada, e fica oculto
# quando não há lugar fora sem cortar nem sobrepor; no anel de dentro da rosca, vai para o furo. Fatia de 0% fica sem
# rótulo. O diâmetro das 2 fica fixo, sem mudar com o bairro nem com os filtros: o domínio de cada traço tem a altura
# do diâmetro, e os textos de fora, os do furo e as linhas que identificam os anéis são anotações, que não tomam
# espaço delas
FATIA_DENTRO = 0.25
BORDA_FATIA = 2
# rosca: margem de cima da legenda de 1 linha (32 px), com as identificações dos anéis logo abaixo, em até 3 linhas,
# porque a de fora quebra acima de LINHA_DO_ANEL caracteres. Diâmetro: o maior que deixa o texto de fora de uma fatia
# pequena junto das 12 horas, ao lado do topo da rosca, centrado na linha de chamada e 2 px abaixo dessas 3 linhas;
# centro em px do topo do gráfico. Espessura: o percentual cabe sem girar em toda fatia do anel de fora, de 5,5% a 100%
TOPO_ROSCA = 34
LINHA_DO_ANEL = 40
DIAMETRO_ROSCA = 245
CENTRO_ROSCA = 224.5
ESPESSURA_ANEL = 43
SEPARACAO_ANEIS = 3
# o plotly não põe linha de chamada no texto de fora nem o segura dentro do gráfico: a posição sai de caixas_de_fora,
# em px a partir do centro, com a largura dos textos medida na fonte dos gráficos (Source Sans, 14 px) e a largura
# dos gráficos na janela de 1.490 px
LARGURA_TEXTO = {"Baía de Todos": 81.61, "os Santos": 56.52, "Central": 42.92, "Atlântico": 53.71}
# percentual mais largo com 4, 5 e 6 caracteres ("4,0%", "40,0%" e "100,0%")
LARGURA_PERCENTUAL = {4: 30.402, 5: 37.36, 6: 43.318}
ALTURA_LINHA = 19
AVANCO_LINHA = 18.2
LARGURA_SEXO = 290.0
LARGURA_RISP = 258.8
LINHA_CHAMADA = 10
# pizza: diâmetro e centro (fração da largura e px do topo do gráfico)
DIAMETRO_PIZZA = 210
CENTRO_PIZZA = (144.4 / LARGURA_RISP, 229)
OPACIDADE_OBSERVADO = 0.8
OPACIDADE_CONTRIBUICAO = 0.88
SEM_BARRA = {"displayModeBar": False}
ESTATICO = {"staticPlot": True}
# alturas dos gráficos da seção de explicação: compensam as linhas de texto de cada cartão, para os
# cartões lado a lado (7c com 7a, 7b com 7d) fecharem juntos na largura das capturas
ALTURAS = {"normal": {"cascata": 427, "enxame": 510, "dependencia": 330, "anos": 361},
           "modelo": {"enxame": 442, "dependencia": 480, "anos": 552},
           "modelo_30": {"enxame": 1300, "dependencia": 480, "anos": 1500},
           "demonstracao": {"enxame": 545, "dependencia": 480, "anos": 552, "agregacao": 420}}

st.set_page_config(page_title="VEX-CVLI, painel mensal (mock)", layout="wide")

# a base tem dados de vítimas: fora de localhost ou com estatísticas de uso ligadas, o mock para
# antes de ler qualquer arquivo
if st.get_option("server.address") not in ("localhost", "127.0.0.1") or st.get_option("browser.gatherUsageStats"):
    st.error("O mock só abre servido em localhost e sem estatísticas de uso. Da raiz do projeto: "
             "./venv/Scripts/python.exe -m streamlit run painel_mensal/app.py --server.address localhost "
             "--browser.gatherUsageStats false --client.toolbarMode minimal")
    st.stop()

# o crédito do mapa de fundo é do MapLibre, fora do alcance da figura; com os 12 px de origem ele fica abaixo
# de 6 pt nas imagens do capítulo. O MapLibre também troca a seta do mouse pela mão de arrastar sobre o mapa;
# a seta fica, e arrastar e dar zoom continuam funcionando. Na barra lateral, a nota "com dados até" sai com
# opacidade 0,6 no tema, 3,6:1 contra o fundo da barra; fica opaca, na cor do texto dos gráficos
# o seletor pega os 2 mapas, inclusive o da exploração com a chave nova que ele ganha a cada clique
st.html("<style>.maplibregl-ctrl-attrib, .maplibregl-ctrl-attrib a { font-size: %dpx !important; }"
        "[class*='st-key-mapa'] .maplibregl-canvas-container, "
        "[class*='st-key-mapa'] .maplibregl-canvas-container:active, "
        "[class*='st-key-mapa'] .maplibregl-canvas { cursor: default !important; }"
        "[data-testid='stSidebar'] [data-testid='stCaptionContainer'] { opacity: 1 !important; color: %s !important; }"
        "</style>" % (rotulos.FONTE, rotulos.COR_TEXTO))


@st.cache_data
def carregar_tabelas():
    return dados.ler_eventos(), dados.ler_base(), dados.atributos_do_modelo()


# cache_resource: o GeoJSON tem 6 MB e não é alterado, então não vale copiar a cada execução
@st.cache_resource
def carregar_geojson(bairros):
    geo = dados.ler_geojson()
    return geo, dados.enquadramento(geo, [dados.poligono(b) for b in bairros], ALTURA_MAPA)


# os 16 polígonos das AISP, a união dos bairros de cada uma, montados uma vez por servidor
@st.cache_resource
def carregar_geojson_aisp(bairros):
    geo_aisp, corrigidos = dados.poligonos_aisp(carregar_geojson(bairros)[0],
                                                dados.aisp_por_bairro(carregar_tabelas()[1]))
    assert not corrigidos, corrigidos
    return geo_aisp


# cache_resource: as saídas da rodada só são lidas, nunca alteradas
@st.cache_resource
def carregar_modelo():
    return modelo.carregar()


@st.cache_resource
def carregar_metricas():
    return modelo.carregar_metricas()


@st.cache_data
def cascata_do_caso(bairro, ano, mes, todos):
    """PNG da cascata do caso, com o modelo do corte fixo; com a chave fechada, 9 barras e os demais somados."""
    r = carregar_modelo()
    caso = modelo.indice(r, bairro, pd.Period(year=ano, month=mes, freq="M"))
    barras = len(r.atributos) if todos else MOSTRADOS
    return visoes_shap.imagem_cascata(r.phi.loc[caso, r.atributos].to_numpy(dtype=float),
                                      float(r.linhas.loc[caso, "valor_base"]), r.valores.loc[caso, r.atributos].tolist(),
                                      r.atributos, (LARGURA_CASCATA, 1.6 + 0.5 * barras),
                                      max_display=None if todos else MOSTRADOS,
                                      n_vitimas=float(r.valores.loc[caso, "n_vitimas_hist12"]))


def registros(n):
    return dados.inteiro_br(n) + (" registro" if n == 1 else " registros")


def vitimas(n):
    return "%d %s" % (n, "vítima" if n == 1 else "vítimas")


def numero_casas(v, casas):
    """Número com as casas fixas e vírgula decimal."""
    return ("%.*f" % (casas, v)).replace(".", ",")


def meio_do_mes(p):
    return p.start_time + pd.Timedelta(days=14)


def manter_escolha(chave, padrao):
    # o segmented_control deixa desmarcar a opção ativa; o controle volta ao padrão
    if st.session_state.get(chave) is None:
        st.session_state[chave] = padrao


def chave_do_mapa(prefixo="mapa"):
    """Chave de um dos 2 mapas (o da exploração, "mapa", e o da contribuição, "mapa_contribuicao"): muda a cada
    clique (aplicar_cliques), e o mapa novo começa sem seleção."""
    g = st.session_state.get("geracao_" + prefixo, 0)
    return prefixo if g == 0 else "%s_%d" % (prefixo, g)


def bairros_clicados(chave, bairros):
    estado = st.session_state.get(chave)
    pontos = estado["selection"]["points"] if estado else []
    return [cd[0] for cd in (p.get("customdata") or [None] for p in pontos) if cd and cd[0] in bairros]


def aplicar_cliques(bairros):
    """Antes dos controles da página. No mapa da exploração, só na agregação por bairro, o bairro clicado entra na
    seleção da exploração, ou sai, se já estava. Depois de qualquer clique o mapa ganha chave nova: a seleção do
    plotly continua na sessão e, com a mesma chave, o mesmo bairro não poderia ser clicado de novo, nem o clique dado
    em outra agregação ficaria sem efeito ao voltar para a de bairro.

    No mapa da contribuição, só na agregação por bairro, o clique troca o caso explicado; o clique numa AISP ou numa
    RISP não muda nada. Depois de qualquer clique, este mapa também ganha chave nova, e a seleção do plotly que fica na
    sessão não desfaz a troca feita em seguida pela lista da barra lateral.
    """
    estado = st.session_state.get(chave_do_mapa())
    if estado and estado["selection"]["points"]:
        if st.session_state.get("agregacao") == "bairro":
            sel = list(st.session_state.get("sel_exploracao", []))
            for b in dict.fromkeys(bairros_clicados(chave_do_mapa(), bairros)):
                if b in sel:
                    sel.remove(b)
                else:
                    sel.append(b)
            st.session_state["sel_exploracao"] = sel
        st.session_state["geracao_mapa"] = st.session_state.get("geracao_mapa", 0) + 1
    chave = chave_do_mapa("mapa_contribuicao")
    estado = st.session_state.get(chave)
    if estado and estado["selection"]["points"]:
        clicado = (bairros_clicados(chave, bairros) or [None])[0]
        if clicado and st.session_state.get("agregacao_contribuicao") == "bairro":
            st.session_state["caso_bairro"] = st.session_state["sel_bairro"] = clicado
        st.session_state["geracao_mapa_contribuicao"] = st.session_state.get("geracao_mapa_contribuicao", 0) + 1


def figura_mapa(tabela, modo, agregacao, selecao, geo, quadro, nomes_sem_bairro, atributo_mapa=None, geo_aisp=None):
    """No modo contribuição, com o modelo (atributo_mapa dado), a coluna valor da tabela é a média do valor
    absoluto da contribuição nas linhas de 2024; sem o modelo, o mapa fica reservado. Na agregação por AISP (no modo
    observado e, com o modelo, no de contribuição), a tabela é a de dados.tabela_aisp e os polígonos são os de
    geo_aisp, a união dos bairros de cada AISP. O contorno preto marca os bairros de `selecao`."""
    chave = "properties." + dados.CHAVE_POLIGONO
    opacidade = OPACIDADE_OBSERVADO if modo == "observado" else OPACIDADE_CONTRIBUICAO
    centro, zoom = quadro
    fonte = dict(size=rotulos.FONTE, color=rotulos.COR_TEXTO)
    barra = dict(tickfont=fonte, thickness=12, lenmode="pixels", len=200, x=0.01, xanchor="left",
                 y=1 - TOPO_LEGENDA / ALTURA_MAPA, yanchor="top", bgcolor="rgba(255,255,255,0.85)")
    if modo == "observado" and agregacao == "aisp":
        # o 1º campo do customdata fica vazio: o clique numa AISP não muda a seleção de bairros
        texto = "AISP " + tabela["aisp"].astype(str) + ": " + tabela["valor"].map(registros)
        fig = px.choropleth_map(
            tabela.assign(texto=texto, vazio=""), geojson=geo_aisp, locations="aisp",
            featureidkey="properties." + dados.CHAVE_AISP, color="valor", color_continuous_scale=ESCALA_VERMELHA,
            range_color=(0, max(1, int(tabela["valor"].max()))), custom_data=["vazio", "texto"],
            map_style=ESTILO_MAPA, zoom=zoom, center=centro, opacity=opacidade, height=ALTURA_MAPA)
        fig.update_traces(hovertemplate="%{customdata[1]}<extra></extra>")
        fig.update_layout(coloraxis_colorbar=dict(title=dict(text="registros<br>na AISP", font=fonte), **barra))
    elif modo == "observado":
        n = tabela["valor"].map(registros)
        texto = tabela["bairro"] + "<br>RISP " + tabela["risp"] + ": " + n if agregacao == "risp" else tabela["bairro"] + ": " + n
        fig = px.choropleth_map(
            tabela.assign(texto=texto), geojson=geo, locations="poligono", featureidkey=chave, color="valor",
            color_continuous_scale=ESCALA_VERMELHA, range_color=(0, max(1, int(tabela["valor"].max()))),
            custom_data=["bairro", "texto"], map_style=ESTILO_MAPA, zoom=zoom, center=centro,
            opacity=opacidade, height=ALTURA_MAPA)
        fig.update_traces(hovertemplate="%{customdata[1]}<extra></extra>")
        # título em duas linhas na agregação por RISP, para a barra caber na largura da baía
        fig.update_layout(coloraxis_colorbar=dict(
            title=dict(text="registros<br>na RISP" if agregacao == "risp" else "registros", font=fonte), **barra))
    elif atributo_mapa is not None and agregacao == "aisp":
        # contribuição por AISP: os polígonos das 16 AISP, os do mapa da exploração; o 1º campo do customdata fica
        # vazio, e o clique numa AISP não muda o caso explicado
        texto = "AISP " + tabela["aisp"].astype(str) + ": " + tabela["valor"].map(visoes_shap.numero_br)
        fig = px.choropleth_map(
            tabela.assign(texto=texto, vazio=""), geojson=geo_aisp, locations="aisp",
            featureidkey="properties." + dados.CHAVE_AISP, color="valor", color_continuous_scale=ESCALA_LARANJA,
            range_color=(0, float(tabela["valor"].max())), custom_data=["vazio", "texto"], map_style=ESTILO_MAPA,
            zoom=zoom, center=centro, opacity=opacidade, height=ALTURA_MAPA)
        fig.update_traces(hovertemplate="%{customdata[1]}<extra></extra>")
        fig.update_layout(separators=",.", coloraxis_colorbar=dict(title=dict(text="média de<br>|contribuição|", font=fonte),
                                                                   **barra))
        fig.add_annotation(text="%s<br>linhas de 2024 (teste),<br>modelo do corte fixo,<br>%s"
                                % (atributo_mapa, rotulos.ESCALA), x=0.99, y=0.99, xref="paper", yref="paper",
                           xanchor="right", yanchor="top", align="left", showarrow=False,
                           bgcolor="rgba(255,255,255,0.92)", borderpad=6, font=fonte)
    elif atributo_mapa is not None:
        v = tabela["valor"].map(visoes_shap.numero_br)
        texto = tabela["bairro"] + "<br>RISP " + tabela["risp"] + ": " + v if agregacao == "risp" else tabela["bairro"] + ": " + v
        fig = px.choropleth_map(
            tabela.assign(texto=texto), geojson=geo, locations="poligono", featureidkey=chave, color="valor",
            color_continuous_scale=ESCALA_LARANJA, range_color=(0, float(tabela["valor"].max())),
            custom_data=["bairro", "texto"], map_style=ESTILO_MAPA, zoom=zoom, center=centro,
            opacity=opacidade, height=ALTURA_MAPA)
        fig.update_traces(hovertemplate="%{customdata[1]}<extra></extra>")
        fig.update_layout(separators=",.", coloraxis_colorbar=dict(title=dict(text="média de<br>|contribuição|", font=fonte),
                                                                   **barra))
        # identificação e escala no canto de cima à direita, sobre Lauro de Freitas, onde não há bairro
        fig.add_annotation(text="%s<br>linhas de 2024 (teste),<br>modelo do corte fixo,<br>%s"
                                % (atributo_mapa, rotulos.ESCALA), x=0.99, y=0.99, xref="paper", yref="paper",
                           xanchor="right", yanchor="top", align="left", showarrow=False,
                           bgcolor="rgba(255,255,255,0.92)", borderpad=6, font=fonte)
    else:
        fig = go.Figure(go.Choroplethmap(
            geojson=geo, locations=tabela["poligono"], z=[0] * len(tabela), featureidkey=chave,
            colorscale=[[0, CINZA_POLIGONO], [1, CINZA_POLIGONO]], showscale=False, marker_opacity=opacidade,
            customdata=[[b, b + "<br>" + RESERVADO] for b in tabela["bairro"]],
            hovertemplate="%{customdata[1]}<extra></extra>", showlegend=False))
        fig.update_layout(map=dict(style=ESTILO_MAPA, zoom=zoom, center=centro), height=ALTURA_MAPA)
        fig.add_annotation(text=RESERVADO, x=0.99, y=0.99, xref="paper", yref="paper",
                           xanchor="right", yanchor="top", showarrow=False, bgcolor="rgba(255,255,255,0.92)",
                           bordercolor="#9aa0a6", borderpad=6, font=dict(size=rotulos.FONTE, color="#374151"))
    # sem isto o plotly esmaece os outros polígonos depois do clique
    fig.update_traces(marker_line_width=0.6, marker_line_color="#ffffff",
                      selected_marker_opacity=opacidade, unselected_marker_opacity=opacidade)
    fig.add_trace(go.Choroplethmap(
        geojson=dados.subconjunto(geo, nomes_sem_bairro), locations=nomes_sem_bairro, z=[0] * len(nomes_sem_bairro),
        featureidkey=chave, colorscale=TRANSPARENTE, showscale=False, marker_line_width=1.5,
        marker_line_color=CINZA_CONTORNO, name="sem registro na base", showlegend=False,
        customdata=[[""] for _ in nomes_sem_bairro], hovertemplate="%{location}: sem registro na base<extra></extra>"))
    contornados = [dados.poligono(b) for b in dados.selecionados(selecao)]
    if contornados:
        fig.add_trace(go.Choroplethmap(
            geojson=dados.subconjunto(geo, contornados), locations=contornados, z=[0] * len(contornados),
            featureidkey=chave, colorscale=TRANSPARENTE, showscale=False,
            marker_line_width=LARGURA_CONTORNO_SELECIONADO, marker_line_color=CONTORNO_SELECIONADO,
            hoverinfo="skip", showlegend=False))
    # a legenda do plotly desenha o símbolo com o preenchimento, que aqui é transparente; por isso a
    # legenda dos polígonos sem cor é uma anotação com um quadrado vazado
    fig.add_annotation(text="<span style='color:%s;font-size:17px'>□</span> sem registro na base" % CINZA_CONTORNO,
                       x=0.99, y=ATRIBUICAO / ALTURA_MAPA, xref="paper", yref="paper", xanchor="right", yanchor="bottom",
                       showarrow=False, bgcolor="rgba(255,255,255,0.9)", borderpad=4,
                       font=dict(size=rotulos.FONTE, color=rotulos.COR_TEXTO))
    # o plotly põe os polígonos logo acima da última camada de água do mapa de fundo; no estilo com nomes
    # ela era a dos nomes de mar, acima das vias, e no sem nomes é a do próprio mar, abaixo delas. Acima de
    # todas, como antes, as vias não cruzam os bairros
    fig.update_traces(below="")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), uirevision="mapa",
                      font=dict(size=rotulos.FONTE, color=rotulos.COR_TEXTO))
    return fig


def figura_serie(serie, periodo, mes_alvo, legenda_estimativa=None, estimativa=None):
    """Série mensal de registros (do bairro do caso, na explicação; da seleção ou de Salvador, na exploração). Com
    estimativa (registros esperados por mês, indexados pelo mês), ela é desenhada; sem ela, com legenda_estimativa,
    a legenda reserva a entrada da estimativa. Com mes_alvo None, sem a marca do mês-alvo."""
    fig = go.Figure()
    for ini, fim, rotulo, cor in FAIXAS_SERIE:
        a, b = max(ini, periodo[0]), min(fim, periodo[1])
        if a > b:
            continue
        x0, x1 = a.start_time, (b + 1).start_time
        fig.add_vrect(x0=x0, x1=x1, fillcolor=cor, opacity=0.6, line_width=0, layer="below")
        # fundo claro para a linha do mês-alvo não cortar o rótulo quando cai dentro da faixa
        fig.add_annotation(x=x0 + (x1 - x0) / 2, y=1, yref="paper", yanchor="top", text=rotulo, showarrow=False,
                           font=dict(size=rotulos.FONTE, color="#5f6368"), bgcolor="rgba(255,255,255,0.8)")
    fig.add_scatter(x=[meio_do_mes(p) for p in serie.index], y=serie.to_numpy(), mode="lines", name="observado",
                    line=dict(color=AZUL, width=2), customdata=[dados.mes_rotulo(p) for p in serie.index],
                    hovertemplate="%{customdata}: %{y}<extra></extra>")
    if estimativa is not None:
        fig.add_scatter(x=[meio_do_mes(p) for p in estimativa.index], y=estimativa.to_numpy(), mode="lines+markers",
                        name=LEGENDA_ESTIMATIVA_DESENHADA, line=dict(color=COR_ESTIMATIVA, width=2, dash="dash"),
                        marker=dict(color=COR_ESTIMATIVA, size=6),
                        customdata=["%s: %s esperados" % (dados.mes_rotulo(p), numero_casas(v, 2))
                                    for p, v in estimativa.items()],
                        hovertemplate="%{customdata}<extra></extra>")
    elif legenda_estimativa:
        fig.add_scatter(x=[None], y=[None], mode="lines", name=legenda_estimativa,
                        line=dict(color="#9aa0a6", width=2, dash="dash"), hoverinfo="skip")
    if mes_alvo is not None and periodo[0] <= mes_alvo <= periodo[1]:
        xm = meio_do_mes(mes_alvo)
        # a anotação vai à parte: add_vline com texto falha em eixo de datas
        fig.add_vline(x=xm, line=dict(color=VERMELHO, width=1.5, dash="dot"))
        lado = "right" if (mes_alvo - periodo[0]).n > (periodo[1] - periodo[0]).n / 2 else "left"
        fig.add_annotation(x=xm, y=1, yref="paper", xanchor=lado, yanchor="bottom",
                           text="mês-alvo " + dados.mes_rotulo(mes_alvo), showarrow=False,
                           font=dict(size=rotulos.FONTE, color=VERMELHO))
    # a legenda fica também quando só há o observado
    fig.update_layout(height=ALTURA_SERIE, margin=dict(l=0, r=0, t=24, b=0), font=dict(size=rotulos.FONTE),
                      showlegend=True, legend=dict(orientation="h", yanchor="top", y=-0.12, x=0, font_size=rotulos.FONTE))
    fig.update_xaxes(range=[periodo[0].start_time, (periodo[1] + 1).start_time], showgrid=False)
    if (periodo[1] - periodo[0]).n >= 24:
        # um rótulo por ano, no meio do ano; a marca fica na virada, e o ano seguinte ao fim do período não aparece
        fig.update_xaxes(dtick="M12", tick0="2018-01-01", tickformat="%Y", ticklabelmode="period", ticks="outside")
    fig.update_yaxes(title_text="registros no mês", rangemode="tozero", dtick=1 if serie.max() <= 6 else None)
    return rotulos.letra_minima(fig)


def escopos_da_figura(totais, nome_selecao):
    """(escopo, nome na legenda e no hover, cor): a seleção, se houver, em AZUL, e Salvador em cinza."""
    return ([("bairro", nome_selecao, AZUL)] if "bairro" in totais else []) + [("cidade", "Salvador", CINZA_CIDADE)]


def figura_turno(dist, totais, nome_selecao, turnos_marcados):
    """As 4 barras aparecem sempre; o filtro de turno só esmaece as desmarcadas, e o denominador
    continua sendo o total dos 4 turnos do período filtrado. O destaque dos turnos marcados fica só nas
    barras: os números de todas têm a cor do texto dos gráficos. Sem seleção, só Salvador."""
    fig = go.Figure()
    maior = 0.0
    for escopo, nome, cor in escopos_da_figura(totais, nome_selecao):
        d = dist[dist["escopo"] == escopo]
        pct = 100 * d["fracao"].to_numpy()
        maior = max(maior, pct.max(initial=0.0))
        marcado = d["turno"].isin(turnos_marcados).tolist()
        fig.add_bar(x=[dados.rotulo_turno(t) for t in d["turno"]], y=pct, marker_color=cor,
                    marker_opacity=[1.0 if m else 0.25 for m in marcado],
                    name="%s (%s)" % (nome, registros(totais[escopo])),
                    # só o número: com o % junto, o rótulo de 14 px passa da largura da barra
                    text=[dados.pct_br(f).rstrip("%") for f in d["fracao"]], textposition="outside", cliponaxis=False,
                    textfont=dict(size=rotulos.FONTE, color=rotulos.COR_TEXTO),
                    customdata=[registros(r) for r in d["registros"]],
                    hovertemplate="%{customdata}, %{text}<extra>" + nome + "</extra>")
    # showlegend: sem seleção há uma série só, e o plotly esconderia a legenda com o total de Salvador
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.06, height=ALTURA_AO_LADO_DO_MAPA,
                      margin=dict(l=0, r=0, t=10, b=0), showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font_size=rotulos.FONTE),
                      font=dict(size=rotulos.FONTE))
    fig.update_xaxes(tickangle=0, tickfont_size=rotulos.FONTE)
    fig.update_yaxes(title_text="% dos 4 turnos no período", ticksuffix="%", range=[0, max(5.0, maior * 1.25)])
    return rotulos.letra_minima(fig)


def figura_idade(dist, totais, nome_selecao):
    """Histograma da idade da vítima em faixas de 10 anos, com as barras sobrepostas e centradas na faixa: Salvador
    atrás, mais larga, em cinza, e a seleção na frente, mais estreita, em AZUL. O rótulo é só o percentual da
    seleção, em AZUL e acima da mais alta das 2 barras, para mostrar de quem é o número; o de Salvador fica no hover.
    Sem seleção, só Salvador, e o rótulo é o dela, na cor do texto dos gráficos. Legenda acima, com a margem de cima
    fixa das distribuições da linha de baixo: a área das barras não muda com o nome da seleção. Série sem registro
    com idade fica com as barras em 0, e sem rótulo."""
    fig = go.Figure()
    faixas = list(dados.FAIXAS_IDADE)
    pct = {}
    # Salvador primeiro, para ficar atrás; a legenda, em ordem inversa, mostra a seleção antes
    for escopo, nome, cor, largura in (("cidade", "Salvador", CINZA_CIDADE, LARGURA_IDADE_CIDADE),
                                       ("bairro", nome_selecao, AZUL, LARGURA_IDADE_BAIRRO)):
        if escopo not in totais:
            continue
        d = dist[dist["escopo"] == escopo]
        pct[escopo] = (100 * d["fracao"]).tolist()
        quebra = "<br>" if escopo == "bairro" and len(nome) > NOME_LONGO else " "
        hover = [[registros(r), dados.pct_br(f).rstrip("%")] for r, f in zip(d["registros"], d["fracao"])]
        fig.add_bar(x=faixas, y=pct[escopo], width=largura, marker_color=cor,
                    name="%s%s(%s)" % (nome, quebra, registros(totais[escopo])), customdata=hover,
                    hovertemplate="%{customdata[0]}, %{customdata[1]}<extra>" + nome + "</extra>")
    rotulado, cor_rotulo = ("bairro", AZUL) if "bairro" in totais else ("cidade", rotulos.COR_TEXTO)
    if totais[rotulado]:
        fracoes = dist.loc[dist["escopo"] == rotulado, "fracao"]
        # o rótulo é um traço de texto à parte, sem hover e fora da legenda, porque a barra da seleção pode ficar
        # abaixo da de Salvador. O marcador, invisível, de 6 px afasta o texto do topo da barra: o plotly desloca o
        # texto pelo raio do marcador, e assim a caixa do texto fica 3 px acima da barra, como nas barras do turno
        fig.add_scatter(x=faixas, y=[max(v) for v in zip(*pct.values())], mode="markers+text",
                        marker=dict(size=6, color="rgba(0,0,0,0)"),
                        text=[dados.pct_br(f).rstrip("%") for f in fracoes], textposition="top center",
                        textfont=dict(size=rotulos.FONTE, color=cor_rotulo), cliponaxis=False, hoverinfo="skip",
                        showlegend=False)
    limite = max(5.0, max(sum(pct.values(), [])) * 1.25)
    marcas = marcas_ate_100(limite)
    fig.update_layout(barmode="overlay", height=ALTURA_AO_LADO_DO_MAPA, margin=dict(l=0, r=0, t=TOPO_DISTRIBUICAO, b=0),
                      showlegend=True, legend=dict(orientation="h", xref="container", yref="container", x=0, y=1, xanchor="left",
                                  yanchor="top", traceorder="reversed", font_size=rotulos.FONTE),
                      font=dict(size=rotulos.FONTE))
    fig.update_xaxes(tickangle=0, tickfont_size=rotulos.FONTE)
    # com as marcas dadas uma a uma, o plotly não põe o ticksuffix; o % vai no texto de cada marca
    fig.update_yaxes(title_text="% com idade informada", range=[0, limite], tickvals=marcas,
                     ticktext=["%d%%" % m for m in marcas])
    return rotulos.letra_minima(fig)


def marcas_ate_100(limite):
    """Marcas do eixo das distribuições, de 0 ao limite e sem passar de 100%: o eixo pode ir além para caber o
    rótulo da barra maior, mas a escala de porcentagem termina em 100. Passo de 1, 2, 5, 10, 20, 25 ou 50, com
    até 4 intervalos."""
    topo = min(limite, 100.0)
    passo = next(p for p in (1, 2, 5, 10, 20, 25, 50) if topo / p <= 4)
    return [passo * i for i in range(int(topo // passo) + 1)]


def figura_distribuicao(categorias, series, titulo_eixo):
    """Barras horizontais da distribuição por natureza, no padrão da do turno: categorias de cima para baixo na
    ordem dada, rótulo só com o número, por fora da barra, legenda acima e eixo de 0 até 1,25 vez o maior valor.
    Cada série: (legenda, nome no hover, cor, índice da categoria de cada barra, centro da barra na categoria em
    unidades da natureza, frações, registros, opacidades)."""
    n = len(categorias)
    escala = n / CATEGORIAS_DA_NATUREZA
    largura = LARGURA_BARRA * escala
    fig = go.Figure()
    maior = 0.0
    for legenda, nome, cor, posicoes, centro, fracoes, contagens, opacidades in series:
        pct = [100 * f for f in fracoes]
        maior = max([maior, *pct])
        fig.add_bar(y=posicoes, x=pct, orientation="h", width=largura, offset=centro * escala - largura / 2,
                    marker_color=cor, marker_opacity=opacidades, name=legenda,
                    text=[dados.pct_br(f).rstrip("%") for f in fracoes], textposition="outside", cliponaxis=False,
                    textfont=dict(size=rotulos.FONTE, color=rotulos.COR_TEXTO),
                    customdata=[registros(r) for r in contagens],
                    hovertemplate="%{customdata}, %{text}<extra>" + nome + "</extra>")
    limite = max(5.0, maior * 1.25)
    marcas = marcas_ate_100(limite)
    # a legenda começa na borda do gráfico, e não na da área das barras, que fica à direita dos nomes das
    # categorias. A margem de cima é fixa na altura de 3 linhas de legenda: com a legenda em 1, 2 ou 3 linhas, que
    # depende da largura da janela e do nome do bairro, a área das barras tem a mesma altura em todos os bairros, e
    # as barras, a mesma espessura
    fig.update_layout(barmode="overlay", height=ALTURA_DISTRIBUICAO, margin=dict(l=0, r=0, t=TOPO_DISTRIBUICAO, b=0),
                      showlegend=True,
                      legend=dict(orientation="h", xref="container", yref="container", x=0, y=1, xanchor="left",
                                  yanchor="top", traceorder="normal", font_size=rotulos.FONTE),
                      font=dict(size=rotulos.FONTE))
    fig.update_yaxes(tickvals=list(range(n)), ticktext=categorias, range=[n - 0.5, -0.5], showgrid=False,
                     zeroline=False, tickfont_size=rotulos.FONTE)
    # com as marcas dadas uma a uma, o plotly não põe o ticksuffix; o % vai no texto de cada marca
    fig.update_xaxes(title_text=titulo_eixo, range=[0, limite], tickvals=marcas, ticktext=["%d%%" % m for m in marcas])
    return rotulos.letra_minima(fig)


def series_bairro_cidade(dist, totais, nome_selecao, opacidades):
    """As 2 séries agrupadas da natureza: em cada categoria, a seleção em cima, em AZUL, e Salvador embaixo, em
    cinza, na ordem da legenda. Sem seleção, só Salvador, com a barra no meio da categoria."""
    series = []
    for escopo, nome, cor in escopos_da_figura(totais, nome_selecao):
        centro = 0.0 if len(totais) == 1 else CENTRO_BAIRRO if escopo == "bairro" else -CENTRO_BAIRRO
        d = dist[dist["escopo"] == escopo]
        quebra = "<br>" if len(nome) > NOME_LONGO else " "
        series.append(("%s%s(%s)" % (nome, quebra, registros(totais[escopo])), nome, cor, list(range(len(d))), centro,
                       d["fracao"].tolist(), d["registros"].tolist(), opacidades))
    return series


def figura_natureza(dist, totais, nome_selecao, naturezas_marcadas):
    """As 4 naturezas aparecem sempre; o filtro de natureza só esmaece as desmarcadas, e o denominador continua
    sendo o total das 4."""
    opacidades = [1.0 if n in naturezas_marcadas else 0.25 for n in dados.NATUREZAS]
    return figura_distribuicao([EIXO_NATUREZA[n] for n in dados.NATUREZAS],
                               series_bairro_cidade(dist, totais, nome_selecao, opacidades),
                               "% das 4 naturezas no período")


def direcao(angulo):
    """Unitário do centro para o ângulo, em graus no sentido horário a partir das 12 horas, com y para baixo."""
    a = math.radians(angulo)
    return math.sin(a), -math.cos(a)


def encosta(caixa, r, folga=2):
    """A caixa (x0, y0, x1, y1, em px a partir do centro) chega a menos de `folga` px do círculo de raio r?"""
    x = min(max(0, caixa[0]), caixa[2])
    y = min(max(0, caixa[1]), caixa[3])
    return math.hypot(x, y) < r + folga


def sobrepoe(a, b, folga=2):
    return not (a[2] + folga <= b[0] or b[2] + folga <= a[0] or a[3] + folga <= b[1] or b[3] + folga <= a[1])


def cruza(linha, caixa, folga=2):
    """O segmento (p, q) passa pela caixa alargada de `folga` px? Recorte de Liang-Barsky."""
    (x0, y0), (x1, y1) = linha
    t0, t1 = 0.0, 1.0
    for p, q in ((x0 - x1, x0 - caixa[0] + folga), (x1 - x0, caixa[2] + folga - x0),
                 (y0 - y1, y0 - caixa[1] + folga), (y1 - y0, caixa[3] + folga - y0)):
        if p == 0:
            if q < 0:
                return False
        elif p < 0:
            t0 = max(t0, q / p)
        else:
            t1 = min(t1, q / p)
    return t0 < t1


def caixas_de_fora(itens, r, limites, ombro_primeiro=False):
    """Lugar do texto de fora de cada fatia, em px a partir do centro, com y para baixo. Na direção da fatia, a
    LINHA_CHAMADA da borda e com o canto mais perto dela voltado para a fatia; se sair dos limites, encostar no
    círculo ou em outro texto, acima ou abaixo dele e depois à esquerda ou à direita, do lado da fatia; por fim, no
    ombro: ao lado do círculo, a 3 px dele, na altura do ponto da fatia, deslizando na vertical para caber nos
    limites. Nas outras 4 posições o texto pode deslizar na horizontal. A linha de chamada (da borda da fatia ao ponto
    da caixa mais perto dela, como em texto_de_fora) não pode cruzar outro texto, nem a de outro texto cruzar a caixa.
    Com ombro_primeiro, o ombro vem antes das outras: na rosca, o texto de fora é de fatia perto das 12 horas, e
    acima dela ficam as identificações dos anéis. itens: [(ângulo do meio, largura, altura)]; limites: (x0, y0, x1,
    y1). Devolve a caixa de cada um, ou None quando nenhuma posição fica livre: esse texto fica oculto e não ocupa
    lugar."""
    xmin, ymin, xmax, ymax = limites
    feitas, linhas, saida = [], [], []
    for angulo, w, h in itens:
        ux, uy = direcao(angulo)
        qx, qy = (r + LINHA_CHAMADA) * ux, (r + LINHA_CHAMADA) * uy
        x0 = qx if ux > 0.25 else (qx - w if ux < -0.25 else qx - w / 2)
        y0 = qy - h if uy < -0.25 else (qy if uy > 0.25 else qy - h / 2)
        px, py = r * ux, r * uy
        candidatas = [(x0, y0), (px - w / 2, -r - LINHA_CHAMADA - h if uy < 0 else r + LINHA_CHAMADA),
                      (-r - LINHA_CHAMADA - w if ux < 0 else r + LINHA_CHAMADA, py - h / 2)]
        # ombro a 3 px do círculo, porque a borda branca das fatias passa 1 px para fora dele. O lado sai do ângulo, e
        # não de ux: a fatia pequena do fim da volta fica perto dos 360 graus, à esquerda, e a do começo, à direita
        oy = min(max((r + 1) * uy - h / 2, ymin), ymax - h)
        perto = 0 if oy <= 0 <= oy + h else min(abs(oy), abs(oy + h))
        dx = math.sqrt(max(0.0, (r + 3) ** 2 - perto ** 2))
        candidatas.insert(0 if ombro_primeiro else len(candidatas), (-dx - w if angulo > 180 else dx, oy))
        ponta = ((r + 1) * ux, (r + 1) * uy)
        escolhida = None
        for cx0, cy0 in candidatas:
            cx0 = min(max(cx0, xmin), xmax - w)
            c = (cx0, cy0, cx0 + w, cy0 + h)
            if cy0 < ymin or cy0 + h > ymax or encosta(c, r) or any(sobrepoe(c, f) for f in feitas):
                continue
            x, y, _, _ = ponto_mais_perto(c, ponta)
            if any(cruza((ponta, (x, y)), f) for f in feitas) or any(cruza(l, c) for l in linhas):
                continue
            escolhida = c
            linhas.append((ponta, (x, y)))
            feitas.append(c)
            break
        saida.append(escolhida)
    return saida


def caixa_do_texto(nome, f):
    """Largura e altura do texto de uma fatia: as linhas do nome (nenhuma, na rosca) e o percentual embaixo."""
    return (max([LARGURA_TEXTO[l] for l in nome] + [LARGURA_PERCENTUAL[len(dados.pct_br(f))]]),
            ALTURA_LINHA + AVANCO_LINHA * len(nome))


def ponto_mais_perto(caixa, alvo):
    """Dos 8 pontos da borda da caixa (cantos e meios dos lados), o mais perto do alvo, com as âncoras dele."""
    x0, y0, x1, y1 = caixa
    pontos = [(x, y, xa, ya) for x, xa in ((x0, "left"), ((x0 + x1) / 2, "center"), (x1, "right"))
              for y, ya in ((y0, "top"), ((y0 + y1) / 2, "middle"), (y1, "bottom"))
              if (xa, ya) != ("center", "middle")]
    return min(pontos, key=lambda p: math.hypot(p[0] - alvo[0], p[1] - alvo[1]))


def lugares_de_fora(r, fracoes, de_fora, linhas, limites, ombro_primeiro=False):
    """Onde fica o texto de cada fatia marcada em de_fora: [(ângulo do meio, texto, caixa)], com a caixa em px a
    partir do centro, y para baixo, ou None quando não há lugar (texto oculto, só no hover). Fatia de 0% não entra:
    fica sem rótulo. linhas: as do nome de cada fatia, sem o percentual; limites e ombro_primeiro: os de
    caixas_de_fora."""
    itens, textos, inicio = [], [], 0.0
    for f, fora, nome in zip(fracoes, de_fora, linhas):
        if fora and f > 0:
            itens.append((360 * (inicio + f / 2), *caixa_do_texto(nome, f)))
            textos.append("<br>".join(nome + [dados.pct_br(f)]))
        inicio += f
    caixas = caixas_de_fora(itens, r, limites, ombro_primeiro)
    return [(angulo, texto, c) for (angulo, _, _), texto, c in zip(itens, textos, caixas)]


def texto_de_fora(fig, centro, r, lugares):
    """Desenha os textos de lugares_de_fora que acharam lugar, cada um com a linha de chamada da borda da fatia até o
    ponto mais perto do texto. centro: (x, y) em fração da área do gráfico."""
    for angulo, texto, c in lugares:
        if c is None:
            continue
        ux, uy = direcao(angulo)
        # a linha sai 1 px fora do círculo, além da borda branca das fatias
        bx, by = (r + 1) * ux, (r + 1) * uy
        x, y, xa, ya = ponto_mais_perto(c, (bx, by))
        fig.add_shape(type="line", xref="paper", yref="paper", xsizemode="pixel", ysizemode="pixel",
                      xanchor=centro[0], yanchor=centro[1], x0=bx, y0=-by, x1=x, y1=-y,
                      line=dict(color=rotulos.COR_TEXTO, width=1))
        fig.add_annotation(xref="paper", yref="paper", x=centro[0], y=centro[1], xshift=x, yshift=-y, xanchor=xa,
                           yanchor=ya, text=texto, showarrow=False, align="center", borderpad=0,
                           font=dict(size=rotulos.FONTE, color=rotulos.COR_TEXTO))


def lugares_no_furo(r, furo, fracoes, dentro):
    """No anel de dentro da rosca, o percentual que não cabe na fatia vai para o centro do furo: [(ângulo do meio,
    texto, caixa centrada, ponto da fatia)], com o ponto 1 px para dentro da borda do furo, na direção da fatia. Só
    uma fatia por vez pode precisar do furo, porque a outra tem 50% ou mais e cabe dentro. Fatia de 0% não entra."""
    saida, inicio = [], 0.0
    for f, cabe in zip(fracoes, dentro):
        if f > 0 and not cabe:
            angulo = 360 * (inicio + f / 2)
            w, h = caixa_do_texto([], f)
            ux, uy = direcao(angulo)
            borda = furo * r - 1
            saida.append((angulo, dados.pct_br(f), (-w / 2, -h / 2, w / 2, h / 2), (borda * ux, borda * uy)))
        inicio += f
    return saida


def texto_no_furo(fig, centro, lugares):
    """Desenha os textos de lugares_no_furo, em COR_TEXTO, com uma linha curta do ponto da caixa mais perto da fatia
    até ela."""
    for _, texto, c, (bx, by) in lugares:
        x, y, _, _ = ponto_mais_perto(c, (bx, by))
        fig.add_shape(type="line", xref="paper", yref="paper", xsizemode="pixel", ysizemode="pixel",
                      xanchor=centro[0], yanchor=centro[1], x0=x, y0=-y, x1=bx, y1=-by,
                      line=dict(color=rotulos.COR_TEXTO, width=1))
        fig.add_annotation(xref="paper", yref="paper", x=centro[0], y=centro[1], xanchor="center", yanchor="middle",
                           text=texto, showarrow=False, borderpad=0,
                           font=dict(size=rotulos.FONTE, color=rotulos.COR_TEXTO))


def cabe_dentro(fracoes, r, furo, caixas):
    """O texto de cada fatia (caixas: largura e altura de cada um) cabe nela na horizontal, com a letra cheia? É a
    conta do plotly.js 3.7.0, o do Streamlit, para o texto de dentro com insidetextorientation horizontal: o círculo
    inscrito na fatia e os pontos em que ela cruza 12, 3, 6 ou 9 horas, com a folga de 3 px do plotly; cabe se algum
    dá escala 1 ou mais. A largura do percentual é a do mais largo com o mesmo número de caracteres, então o que cabe
    aqui cabe no plotly. Fatia de 0% não cabe texto."""
    anel = 1 - furo
    saida, inicio = [], 0.0
    for f, (w, h) in zip(fracoes, caixas):
        a0, a1 = 2 * math.pi * inicio, 2 * math.pi * (inicio + f)
        inicio += f
        if f == 0:
            saida.append(False)
            continue
        # o plotly usa o círculo inteiro na pizza de uma fatia só
        inscrito = 1 if f == 1 and furo == 0 else min(1 / (1 + 1 / math.sin(math.pi * min(f, 0.5))), anel / 2)
        escalas = [inscrito * 2 * r / math.hypot(w, h)]
        for meio, tangente in ((0, True), (1, True), (1.5, False), (0.5, False)):
            for k in (4, 2, 0, -2, -4):
                angulo = math.pi * (k + meio)
                if a0 < angulo < a1:
                    a = h / w if tangente else w / h
                    q = a + 1 / (2 * math.tan(min(angulo - a0, a1 - angulo)))
                    d = math.sqrt(q * q + 0.5) + q
                    s = (r - 6) * min(1 / d if d else math.inf, anel / (math.sqrt(a * a + anel / 2) + a))
                    escalas.append(2 * s / (w if tangente else h))
        saida.append(max(escalas) >= 1)
    return saida


def fatias(fig, nomes, contagens, fracoes, cores, cores_texto, textos, dominio, furo, hover, extra, dentro):
    """Uma pizza ou um anel: fatias na ordem dada, a partir das 12 horas, no sentido horário, com borda branca. O
    texto das fatias marcadas em dentro vai dentro, sempre na horizontal; o das outras sai em texto_de_fora ou
    texto_no_furo, ou fica só no hover. A letra não diminui (uniformtext em legenda_e_layout)."""
    pct = [dados.pct_br(f) for f in fracoes]
    fig.add_pie(labels=nomes, values=contagens, sort=False, direction="clockwise", rotation=0, hole=furo,
                marker=dict(colors=cores, line=dict(color="white", width=BORDA_FATIA)), text=textos, textinfo="text",
                textposition=["inside" if d else "none" for d in dentro], insidetextorientation="horizontal",
                insidetextfont=dict(size=rotulos.FONTE, color=cores_texto),
                customdata=[[hover[i], registros(n), p] for i, (n, p) in enumerate(zip(contagens, pct))],
                hovertemplate="%{customdata[0]}: %{customdata[1]}, %{customdata[2]}<extra>" + extra + "</extra>",
                showlegend=False, domain=dominio)


def fatia_vazia(fig, dominio, furo):
    """Sem registro no filtro: um anel (ou disco) inteiro no cinza do espaço reservado, sem texto e sem hover. A borda
    branca é a das fatias, que come 1 px de cada lado: sem ela, o vazio sairia maior que o desenhado."""
    fig.add_pie(values=[1], hole=furo, sort=False,
                marker=dict(colors=[FUNDO_RESERVADO], line=dict(color="white", width=BORDA_FATIA)),
                textinfo="none", hoverinfo="skip", showlegend=False, domain=dominio)


def legenda_e_layout(fig, entradas, topo):
    """Legenda que não depende das fatias, como as das barras: um traço de barras vazio por entrada, com os eixos
    escondidos e sem clique, porque esconder o traço vazio não muda nada no gráfico. O uniformtext em hide esconde
    texto de dentro que só caberia com letra menor que rotulos.FONTE: cabe_dentro faz a mesma conta do plotly, então
    isso não deve acontecer; se acontecer, o texto some em vez de sair da fatia."""
    for nome, cor in entradas:
        fig.add_bar(x=[None], y=[None], name=nome, marker_color=cor, hoverinfo="skip")
    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True)
    fig.update_layout(height=ALTURA_DISTRIBUICAO, margin=dict(l=0, r=0, t=topo, b=0),
                      legend=dict(orientation="h", xref="container", yref="container", x=0, y=1, xanchor="left",
                                  yanchor="top", traceorder="normal", font_size=rotulos.FONTE, itemclick=False,
                                  itemdoubleclick=False),
                      uniformtext=dict(minsize=rotulos.FONTE, mode="hide"), font=dict(size=rotulos.FONTE))
    return rotulos.letra_minima(fig)


def linha_do_anel(posicao, nome, total):
    """Identificação de um anel, abaixo da legenda; com a linha inteira acima de LINHA_DO_ANEL caracteres, o número
    de registros vai para a linha de baixo."""
    inicio, fim = "%s: %s" % (posicao, nome), "(%s)" % registros(total)
    return inicio + (" " if len(inicio) + 1 + len(fim) <= LINHA_DO_ANEL else "<br>") + fim


def figura_sexo(dist, totais, nome_selecao):
    """Rosca de 2 anéis com o mesmo centro e a mesma espessura, a seleção de bairros por fora e Salvador por dentro,
    nas cores do gênero da vítima; o registro sem sexo informado fica fora. No anel de fora, o percentual que não cabe
    na fatia sai com linha de chamada (e fica só no hover se não há lugar); no de dentro, vai para o centro do furo,
    com uma linha curta até a fatia. Sem seleção, só Salvador, no anel de fora, identificado só pelo nome."""
    fig = go.Figure()
    h = ALTURA_DISTRIBUICAO - TOPO_ROSCA
    yc = 1 - (CENTRO_ROSCA - TOPO_ROSCA) / h
    externo = DIAMETRO_ROSCA / 2
    cores = [COR_MASCULINO, COR_FEMININO]
    categorias = [EIXO_SEXO[s] for s in dados.SEXOS]
    # o texto de fora fica abaixo das 3 linhas que as identificações ocupam quando a de fora quebra
    fim_das_linhas = TOPO_ROSCA + ALTURA_LINHA + 2 * AVANCO_LINHA
    limites = (-LARGURA_SEXO / 2 + 1, fim_das_linhas + 2 - CENTRO_ROSCA, LARGURA_SEXO / 2 - 1,
               ALTURA_DISTRIBUICAO - 1 - CENTRO_ROSCA)
    linhas = []
    aneis = ((("bairro", nome_selecao, "por fora", externo),
              ("cidade", "Salvador", "por dentro", externo - ESPESSURA_ANEL - SEPARACAO_ANEIS))
             if "bairro" in totais else (("cidade", "Salvador", None, externo),))
    for escopo, nome, posicao, r in aneis:
        furo = 1 - ESPESSURA_ANEL / r
        dominio = dict(x=[0.5 - r / LARGURA_SEXO, 0.5 + r / LARGURA_SEXO], y=[yc - r / h, yc + r / h])
        linhas.append(linha_do_anel(posicao, nome, totais[escopo]) if posicao
                      else "%s (%s)" % (nome, registros(totais[escopo])))
        if totais[escopo] == 0:
            fatia_vazia(fig, dominio, furo)
            continue
        fr = dist.loc[dist["escopo"] == escopo, "fracao"].tolist()
        dentro = cabe_dentro(fr, r, furo, [caixa_do_texto([], f) for f in fr])
        fatias(fig, categorias, dist.loc[dist["escopo"] == escopo, "registros"].tolist(), fr, cores,
               [rotulos.COR_TEXTO, "white"], [dados.pct_br(f) for f in fr], dominio, furo, categorias, nome, dentro)
        if r == externo:
            texto_de_fora(fig, (0.5, yc), r, lugares_de_fora(r, fr, [not c for c in dentro], [[] for _ in fr],
                                                             limites, ombro_primeiro=True))
        else:
            texto_no_furo(fig, (0.5, yc), lugares_no_furo(r, furo, fr, dentro))
    fig.add_annotation(xref="paper", yref="paper", x=0, y=1, xanchor="left", yanchor="top", align="left",
                       text="<br>".join(linhas), showarrow=False, borderpad=0,
                       font=dict(size=rotulos.FONTE, color=rotulos.COR_TEXTO))
    return legenda_e_layout(fig, list(zip(categorias, cores)), TOPO_ROSCA)


def figura_risp(dist, risps_da_selecao, nome_selecao):
    """Pizza das 3 RISP em Salvador inteira, na ordem de rotulos.ORDEM_RISP, a partir das 12 horas: as RISP que têm
    bairro da seleção em AZUL e as outras em cinza; sem seleção, todas em cinza e sem legenda. O nome e o percentual
    ficam dentro da fatia a partir de FATIA_DENTRO, quando cabem; senão, fora, com linha de chamada, ou só no hover
    quando não há lugar. A legenda é a das barras de antes, sem as fatias."""
    fig = go.Figure()
    h = ALTURA_DISTRIBUICAO - TOPO_DISTRIBUICAO
    r = DIAMETRO_PIZZA / 2
    xc, cy = CENTRO_PIZZA
    yc = 1 - (cy - TOPO_DISTRIBUICAO) / h
    dominio = dict(x=[max(0.0, 2 * xc - 1), min(1.0, 2 * xc)], y=[yc - r / h, yc + r / h])
    if int(dist["registros"].sum()) == 0:
        fatia_vazia(fig, dominio, 0)
    else:
        fr = dist["fracao"].tolist()
        do_bairro = [k in risps_da_selecao for k in dist["risp"]]
        nomes = [EIXO_RISP[k].split("<br>") for k in dist["risp"]]
        cabe = cabe_dentro(fr, r, 0, [caixa_do_texto(n, f) for n, f in zip(nomes, fr)])
        dentro = [f >= FATIA_DENTRO and c for f, c in zip(fr, cabe)]
        fatias(fig, list(dist["risp"]), dist["registros"].tolist(), fr,
               [AZUL if b else CINZA_CIDADE for b in do_bairro],
               ["white" if b else rotulos.COR_TEXTO for b in do_bairro],
               [EIXO_RISP[k] + "<br>" + dados.pct_br(f) for k, f in zip(dist["risp"], fr)], dominio, 0,
               [rotulos.NOMES_RISP[k] for k in dist["risp"]], "", dentro)
        cx = xc * LARGURA_RISP
        texto_de_fora(fig, (xc, yc), r,
                      lugares_de_fora(r, fr, [not d for d in dentro], nomes,
                                      (-cx + 1, TOPO_DISTRIBUICAO + 2 - cy, LARGURA_RISP - cx - 1,
                                       ALTURA_DISTRIBUICAO - 1 - cy)))
    entradas = []
    if nome_selecao:
        quebra = "<br>" if len(nome_selecao) > NOME_LONGO else " "
        # "RISP de Itapuã" com um bairro; "RISP da seleção (N bairros)" com mais
        entradas.append(("RISP " + ("da" if nome_selecao.startswith("seleção") else "de") + quebra + nome_selecao,
                         AZUL))
        if len(risps_da_selecao) < len(dist):
            entradas.append(("outras RISP", CINZA_CIDADE))
    return legenda_e_layout(fig, entradas, TOPO_DISTRIBUICAO)


def figura_reservada(altura, titulo_x=None, titulo_y=None, zero_em=None):
    """Espaço reservado: só os eixos com os títulos e a frase; legenda e marca do caso só entram com o
    gráfico desenhado."""
    fig = go.Figure()
    fig.update_xaxes(title_text=titulo_x, range=[-1, 1], showticklabels=False, showgrid=False,
                     zeroline=zero_em == "x", zerolinecolor="#b8bec7")
    fig.update_yaxes(title_text=titulo_y, range=[-1, 1], showticklabels=False, showgrid=False,
                     zeroline=zero_em == "y", zerolinecolor="#b8bec7")
    fig.add_annotation(text=RESERVADO, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, bgcolor="rgba(255,255,255,0.92)", borderpad=8,
                       font=dict(size=rotulos.FONTE, color=rotulos.COR_TEXTO))
    fig.update_layout(height=altura, margin=dict(l=0, r=0, t=8, b=0), plot_bgcolor=FUNDO_RESERVADO, showlegend=False,
                      font=dict(size=rotulos.FONTE))
    return rotulos.letra_minima(fig)


def papel_da_linha(mes_alvo):
    """Papel da linha do mês-alvo no corte fixo, pelo ano: do treino, da validação ou do teste."""
    if mes_alvo.year <= ULTIMO_ANO_DE_TREINO:
        return "do treino"
    if mes_alvo.year < ANO_DE_TESTE:
        return "da validação"
    return "do teste"


def identificacao_local(mes_alvo):
    """Modelo e papel da linha do mês-alvo no corte fixo, no cartão da explicação local sem o modelo."""
    return "Modelo do corte fixo. Linha %s." % papel_da_linha(mes_alvo)


def faixa_de_contexto(bairro, mes_alvo, linha):
    """O caso numa linha, no topo da aba de explicação: bairro e mês-alvo, papel da linha no corte fixo, modelo,
    observado no mês (quando a base o tem) e o tamanho da janela do perfil."""
    partes = ["%s, %s" % (bairro, dados.mes_rotulo(mes_alvo)), "linha " + papel_da_linha(mes_alvo),
              "modelo do corte fixo"]
    if pd.notna(linha["target_cvli"]):
        partes.append("observado: " + registros(int(linha["target_cvli"])))
    partes.append("perfil sobre " + vitimas(int(linha["n_vitimas_hist12"])))
    return " · ".join(partes)


def rotulo_atributo(atributo):
    return ROTULO_TOTAL if atributo == modelo.TOTAL else atributo


def linha_do_contorno(bairro, mes_alvo, ano_das_linhas, nome_das_linhas):
    """Nomeia o caso marcado com contorno preto no resumo global, que só mostra as linhas de um ano."""
    caso = "%s, %s" % (bairro, dados.mes_rotulo(mes_alvo))
    if mes_alvo.year == ano_das_linhas:
        return "Contorno preto: %s." % caso
    return "Sem contorno: %s fica fora das %s." % (caso, nome_das_linhas)


def linhas_do_caso(linha, rotulo_mes):
    """Observado no mês-alvo e tamanho da janela do perfil, lidos da linha da base; os dois modos usam
    esta função (na demonstração, a linha é da base sintética)."""
    st.markdown("Observado em %s: %s." % (rotulo_mes, registros(int(linha["target_cvli"]))))
    st.markdown("Perfil do bairro calculado sobre %s dos 12 meses anteriores." % vitimas(int(linha["n_vitimas_hist12"])))


@st.cache_resource
def carregar_demonstracao():
    t, phi, base, _ = demonstracao.modelo_e_contribuicoes()
    return t, phi, base, demonstracao.ordem_global(t, phi), demonstracao.imagem_cascata(t, phi, base)


def secao_demonstracao():
    """As 4 visões de explicação com o modelo de demonstração: só dado sintético, nada da seleção real."""
    t, phi, _, ordem, cascata = carregar_demonstracao()
    alturas = ALTURAS["demonstracao"]
    bairro_exemplo, mes_exemplo = demonstracao.BAIRRO_EXEMPLO, demonstracao.MES_EXEMPLO
    st.subheader(demonstracao.TITULO_SECAO, anchor=False)
    col_7c, col_7a = st.columns(2, gap="medium")
    with col_7c, st.container(border=True):
        st.markdown("##### Explicação local: %s, %s" % (bairro_exemplo, dados.mes_rotulo(mes_exemplo)))
        st.markdown(demonstracao.identificacao_local(mes_exemplo))
        # a cascata é uma figura do shap (matplotlib), mostrada na largura do cartão
        with st.container(key="cascata"):
            st.image(cascata, width="stretch")
        linhas_do_caso(dados.linha_da_base(t, bairro_exemplo, mes_exemplo), dados.mes_rotulo(mes_exemplo))
    with col_7a, st.container(border=True):
        st.markdown("##### Resumo global, linhas do teste de 2024")
        st.plotly_chart(demonstracao.figura_enxame(t, phi, ordem, alturas["enxame"]), key="enxame", config=SEM_BARRA)
        st.markdown(linha_do_contorno(bairro_exemplo, mes_exemplo, demonstracao.ANO_EXPLICADO, "linhas de 2024"))

    st.subheader("Como a contribuição varia", anchor=False)
    col_7b, col_7d = st.columns(2, gap="medium")
    with col_7b, st.container(border=True):
        st.markdown("##### Dependência, linhas do teste de 2024")
        abas = demonstracao.ABAS_DEPENDENCIA
        for i, (aba, nome) in enumerate(zip(st.tabs(abas), abas)):
            with aba:
                st.plotly_chart(demonstracao.figura_dependencia(t, phi, nome, alturas["dependencia"]),
                                key="dependencia_%d" % i, config=SEM_BARRA)
                st.markdown(linha_do_contorno(bairro_exemplo, mes_exemplo, demonstracao.ANO_EXPLICADO, "linhas de 2024"))
    with col_7d, st.container(border=True):
        st.markdown("##### Mudança entre anos: 2019 e 2021 no treino, 2024 no teste")
        st.plotly_chart(demonstracao.figura_anos(t, phi, ordem, alturas["anos"]), key="anos", config=SEM_BARRA)


def secao_agregacao():
    """Agregação espacial da demonstração: média de |contribuição| por região, em barras, e não no mapa,
    porque os bairros fictícios não têm polígono. Fica fora de linhas_bc para não mudar as outras capturas."""
    t, phi, _, ordem, _ = carregar_demonstracao()
    st.subheader(demonstracao.TITULO_AGREGACAO, anchor=False)
    with st.container(border=True):
        st.markdown("##### " + demonstracao.CARTAO_AGREGACAO)
        st.markdown(demonstracao.IDENTIFICACAO_LINHAS)
        st.plotly_chart(demonstracao.figura_agregacao(t, phi, ordem, ALTURAS["demonstracao"]["agregacao"]),
                        key="agregacao_regiao", config=SEM_BARRA)


def contorno_na_dependencia(r, caso, atributo, bairro, mes_alvo):
    """Como linha_do_contorno, mais o caso de janela vazia, que não aparece nas abas de proporção."""
    if mes_alvo.year == ANO_DE_TESTE and atributo != "cvli_trend" and r.valores.loc[caso, "n_vitimas_hist12"] == 0:
        return "Sem contorno: %s, %s tem a janela vazia." % (bairro, dados.mes_rotulo(mes_alvo))
    return linha_do_contorno(bairro, mes_alvo, ANO_DE_TESTE, "linhas do teste, 2024")


def secao_modelo(r, bairro, mes_alvo, linha, rotulo_alvo):
    """A faixa de contexto do caso e as 4 visões de explicação com o modelo do corte fixo. Enxame, dependência e
    mapa usam as linhas de 2024; a chave abre os 30 atributos no enxame, na mudança entre anos e na cascata. A
    segunda chave põe o bairro num painel com eixo próprio no enxame e na mudança entre anos; a cascata soma até a
    estimativa e fica igual."""
    with st.container(key="explicacao_1"):
        with st.container(border=True, key="faixa"):
            st.markdown(faixa_de_contexto(bairro, mes_alvo, linha))
        todos = st.toggle("ver os 30 atributos", key="ver_30")
        proprio = st.toggle("bairro em escala própria", key="bairro_proprio")
        separado = "bairro" if proprio else None
        ordem = modelo.ordem_global(r)
        mostrados = ordem if todos else ordem[:MOSTRADOS]
        alturas = ALTURAS["modelo_30" if todos else "modelo"]
        caso = modelo.indice(r, bairro, mes_alvo)
        nome_caso = "%s, %s" % (bairro, rotulo_alvo)
        linhas = modelo.no_ano_explicado(r)
        col_7c, col_7a = st.columns(2, gap="medium")
        with col_7c, st.container(border=True, key="cartao_local"):
            st.markdown("##### Explicação local: %s, %s" % (bairro, rotulo_alvo))
            with st.container(key="cascata", horizontal_alignment="center"):
                st.image(cascata_do_caso(bairro, mes_alvo.year, mes_alvo.month, todos),
                         width=visoes_shap.LARGURA_CASCATA_TELA)
        with col_7a, st.container(border=True, key="cartao_global"):
            st.markdown("##### Resumo global, linhas do teste de 2024")
            st.plotly_chart(visoes_shap.figura_enxame(r.valores, r.phi, mostrados, linhas, caso, nome_caso,
                                                      alturas["enxame"], separado=separado), key="enxame",
                            config=SEM_BARRA)
            st.markdown(linha_do_contorno(bairro, mes_alvo, ANO_DE_TESTE, "linhas do teste, 2024"))
            # na largura da cascata, a linha quebra igual em qualquer janela, e os 2 cartões fecham juntos também
            # numa janela larga
            with st.container(width=visoes_shap.LARGURA_CASCATA_TELA):
                st.markdown(TEXTO_CINZAS)

    with st.container(key="explicacao_2"):
        st.subheader("Como a contribuição varia", anchor=False)
        col_7b, col_7d = st.columns(2, gap="medium")
        with col_7b, st.container(border=True, key="cartao_dependencia"):
            st.markdown("##### Dependência, linhas do teste de 2024")
            for i, (aba, nome) in enumerate(zip(st.tabs(list(ABAS_DEPENDENCIA)), ABAS_DEPENDENCIA)):
                atributo = ATRIBUTO_DA_ABA[nome]
                # nas proporções, as linhas de janela vazia (n_vitimas_hist12 = 0) ficam de fora
                desenhadas = linhas if atributo == "cvli_trend" else linhas & (r.valores["n_vitimas_hist12"] > 0)
                with aba:
                    st.plotly_chart(visoes_shap.figura_dependencia(r.valores, r.phi, atributo, desenhadas,
                                                                   r.linhas["risp"], REGIOES_RISP, caso, nome_caso,
                                                                   alturas["dependencia"], "valor de " + atributo),
                                    key="dependencia_%d" % i, config=SEM_BARRA)
                    st.markdown(contorno_na_dependencia(r, caso, atributo, bairro, mes_alvo))
        with col_7d, st.container(border=True, key="cartao_anos"):
            st.markdown("##### Mudança entre anos: 2019 e 2021 no treino, 2024 no teste")
            st.plotly_chart(visoes_shap.figura_anos(r.linhas["ano_alvo"], r.phi, mostrados, alturas["anos"],
                                                    separado=separado), key="anos", config=SEM_BARRA)


def secao_reservada(bairro, mes_alvo, linha, rotulo_alvo):
    """As 4 visões de explicação sem o modelo: só eixos e a frase de espera."""
    alturas = ALTURAS["normal"]
    st.subheader("Explicação da estimativa", anchor=False)
    col_7c, col_7a = st.columns(2, gap="medium")
    with col_7c, st.container(border=True):
        st.markdown("##### Explicação local: %s, %s" % (bairro, rotulo_alvo))
        st.markdown(identificacao_local(mes_alvo))
        st.plotly_chart(figura_reservada(alturas["cascata"], rotulos.ESCALA, "atributo", zero_em="x"),
                        key="cascata", config=ESTATICO)
        linhas_do_caso(linha, rotulo_alvo)
    with col_7a, st.container(border=True):
        st.markdown("##### Resumo global, linhas do teste de 2024")
        st.plotly_chart(figura_reservada(alturas["enxame"], rotulos.EIXO_CONTRIBUICAO), key="enxame",
                        config=ESTATICO)

    st.subheader("Como a contribuição varia", anchor=False)
    col_7b, col_7d = st.columns(2, gap="medium")
    with col_7b, st.container(border=True):
        st.markdown("##### Dependência, linhas do teste de 2024")
        for i, (aba, nome) in enumerate(zip(st.tabs(list(ABAS_DEPENDENCIA)), ABAS_DEPENDENCIA)):
            with aba:
                st.plotly_chart(figura_reservada(alturas["dependencia"], EIXO_DEPENDENCIA[nome],
                                                 rotulos.EIXO_CONTRIBUICAO, zero_em="y"),
                                key="dependencia_%d" % i, config=ESTATICO)
    with col_7d, st.container(border=True):
        st.markdown("##### Mudança entre anos: 2019 e 2021 no treino, 2024 no teste")
        st.plotly_chart(figura_reservada(alturas["anos"], rotulos.EIXO_CONTRIBUICAO, "atributo", zero_em="x"),
                        key="anos", config=ESTATICO)


def faixa_teste(fig, x0, x1, painel=None):
    """Fundo e rótulo da faixa "teste", como na série mensal."""
    onde = {} if painel is None else dict(row=1, col=painel)
    fig.add_vrect(x0=x0, x1=x1, fillcolor=COR_TESTE, opacity=0.6, line_width=0, layer="below", **onde)
    fig.add_annotation(x=x0 + (x1 - x0) / 2, y=1, xref="x" if painel in (None, 1) else "x%d" % painel,
                       yref="y domain" if painel in (None, 1) else "y%d domain" % painel, yanchor="top", text="teste",
                       showarrow=False, font=dict(size=rotulos.FONTE, color="#5f6368"),
                       bgcolor="rgba(255,255,255,0.8)")


def titulos_dos_paineis(fig):
    fig.update_annotations(font=dict(size=rotulos.FONTE, color=rotulos.COR_TEXTO))


def figura_corte_fixo(anuais):
    """3 painéis (MAE, RMSE, deviance de Poisson) com uma barra horizontal por método, o valor ao lado e o eixo a
    partir de 0; o modelo em AZUL, as referências em cinza."""
    d = anuais[anuais["forma"] == "corte_fixo"].set_index("metodo").loc[list(METODOS)]
    nomes = [NOMES_CORTE_FIXO[k] for k in METODOS]
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.05,
                        subplot_titles=list(NOMES_LONGOS.values()))
    titulos_dos_paineis(fig)
    for i, k in enumerate(NOMES_METRICAS, 1):
        v = d[k].to_numpy(float)
        fig.add_bar(x=v, y=nomes, orientation="h", marker_color=[AZUL] + [CINZA_BARRA] * 4, showlegend=False,
                    text=[numero_casas(x, 3) for x in v], textposition="outside", cliponaxis=False,
                    textfont=dict(size=rotulos.FONTE, color=rotulos.COR_TEXTO), hoverinfo="skip", row=1, col=i)
        fig.update_xaxes(range=[0, v.max() * 1.35], row=1, col=i)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=ALTURA_BARRAS, margin=dict(l=0, r=0, t=30, b=0), separators=",.", bargap=0.3)
    return rotulos.letra_minima(fig)


def linhas_por_metodo(fig, x, valores, painel, legenda, customdata=None):
    """Uma linha com marcador por método, no estilo da janela: o modelo em AZUL, contínuo e mais grosso; cada
    referência em cinza, com traço e marcador próprios."""
    for k in METODOS:
        if k == "modelo":
            linha, marcador = dict(color=AZUL, width=3), dict(symbol="circle", size=8, color=AZUL)
        else:
            traco, simbolo = ESTILO_REFERENCIA[k]
            linha, marcador = dict(color=CINZA_LINHA, width=1.5, dash=traco), dict(symbol=simbolo, size=8, color=CINZA_LINHA)
        extra = dict(customdata=customdata[k], hovertemplate="%{customdata}<extra></extra>") if customdata else dict(
            hoverinfo="skip")
        fig.add_scatter(x=x, y=valores[k], mode="lines+markers", name=NOMES_JANELA[k], line=linha, marker=marcador,
                        showlegend=legenda, legendgroup=k, **extra, **({} if painel is None else dict(row=1, col=painel)))


def figura_janela(anuais):
    """3 painéis (MAE, RMSE, deviance de Poisson) com os anos de 2022 a 2024 no eixo horizontal e uma linha por
    método; 2024 com o fundo da faixa "teste"."""
    d = anuais[anuais["forma"] == "janela"]
    anos = sorted(d["ano"].unique())
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.07, subplot_titles=list(NOMES_LONGOS.values()))
    titulos_dos_paineis(fig)
    for i, k in enumerate(NOMES_METRICAS, 1):
        valores = {m: [float(d[(d["ano"] == a) & (d["metodo"] == m)][k].iloc[0]) for a in anos] for m in METODOS}
        linhas_por_metodo(fig, anos, valores, i, legenda=i == 1)
        faixa_teste(fig, ANO_DO_TESTE - 0.5, ANO_DO_TESTE + 0.5, i)
        fig.update_xaxes(range=[anos[0] - 0.5, anos[-1] + 0.5], tickvals=anos, ticktext=[str(a) for a in anos],
                         showgrid=False, row=1, col=i)
    # a legenda fica abaixo das marcas dos anos, na margem de baixo: a caixa dela tem fundo e cobriria as marcas
    fig.update_layout(height=ALTURA_JANELA, margin=dict(l=0, r=0, t=30, b=80), separators=",.",
                      legend=dict(orientation="h", yanchor="top", y=-0.24, x=0, font_size=rotulos.FONTE))
    return rotulos.letra_minima(fig)


def figura_mensal(mensais, metrica):
    """Os 36 meses da janela, de jan/2022 a dez/2024, com um rótulo por ano e uma linha por método; 2024 com o
    fundo da faixa "teste"."""
    meses = pd.PeriodIndex.from_fields(year=mensais["ano"], month=mensais["mes"], freq="M")
    d = mensais.assign(periodo=meses)
    periodos = sorted(d["periodo"].unique())
    x = [meio_do_mes(p) for p in periodos]
    valores, textos = {}, {}
    for m in METODOS:
        s = d[d["metodo"] == m].set_index("periodo").loc[periodos, metrica]
        valores[m] = s.to_numpy(float)
        textos[m] = ["%s, %s: %s" % (dados.mes_rotulo(p), NOMES_JANELA[m], numero_casas(v, 3)) for p, v in s.items()]
    fig = go.Figure()
    linhas_por_metodo(fig, x, valores, None, legenda=True, customdata=textos)
    inicio_teste = pd.Period(year=ANO_DO_TESTE, month=1, freq="M")
    faixa_teste(fig, inicio_teste.start_time, (periodos[-1] + 1).start_time)
    fig.update_xaxes(range=[periodos[0].start_time, (periodos[-1] + 1).start_time], dtick="M12", tick0="2022-01-01",
                     tickformat="%Y", ticklabelmode="period", ticks="outside", showgrid=False)
    fig.update_yaxes(title_text=NOMES_LONGOS[metrica])
    fig.update_layout(height=ALTURA_MENSAL, margin=dict(l=0, r=0, t=10, b=90), separators=",.",
                      legend=dict(orientation="h", yanchor="top", y=-0.22, x=0, font_size=rotulos.FONTE))
    return rotulos.letra_minima(fig)


def tabela_valores(anuais):
    """As 20 linhas de metricas.csv, corte fixo de 2024 e janela de 2022, 2023 e 2024, com os métodos na ordem."""
    saida = []
    for forma, ano in [("corte_fixo", ANO_DO_TESTE)] + [("janela", a) for a in (2022, 2023, ANO_DO_TESTE)]:
        nomes = NOMES_CORTE_FIXO if forma == "corte_fixo" else NOMES_JANELA
        for m in METODOS:
            r = anuais[(anuais["forma"] == forma) & (anuais["ano"] == ano) & (anuais["metodo"] == m)].iloc[0]
            saida.append({"forma": "corte fixo" if forma == "corte_fixo" else "janela expansiva", "ano": str(ano),
                          "método": nomes[m],
                          "MAE": numero_casas(r["mae"], 3), "RMSE": numero_casas(r["rmse"], 3),
                          "deviance de Poisson": numero_casas(r["deviance"], 3)})
    assert len(saida) == 20
    return pd.DataFrame(saida)


def secao_diagnostico(r, metricas):
    """As métricas do teste de 2024, lidas dos arquivos da Parte 2; o painel não calcula métrica nenhuma. Os cartões
    têm só título: a rodada, as formas de validação e as métricas são explicadas no capítulo."""
    with st.container(border=True, key="cartao_corte_fixo"):
        st.markdown("##### Métricas por corte fixo")
        st.plotly_chart(figura_corte_fixo(metricas.anuais), key="diagnostico_corte_fixo", config=SEM_BARRA)
    with st.container(border=True, key="cartao_janela"):
        st.markdown("##### Métricas por ano na janela expansiva")
        st.plotly_chart(figura_janela(metricas.anuais), key="diagnostico_janela", config=SEM_BARRA)
    with st.container(border=True, key="cartao_mensal"):
        st.markdown("##### Métricas por meses na janela expansiva")
        metrica = st.segmented_control("Métrica", list(NOMES_METRICAS), format_func=NOMES_METRICAS.get,
                                       default="deviance", required=True, key="metrica_mensal")
        st.plotly_chart(figura_mensal(metricas.mensais, metrica), key="diagnostico_mensal", config=SEM_BARRA)
    with st.container(border=True, key="cartao_tabela"):
        st.markdown("##### Erro por forma de validação, ano e método")
        st.table(tabela_valores(metricas.anuais), hide_index=True)


eventos, base, atributos = carregar_tabelas()
rodada = carregar_modelo()
bairros = dados.ordenar_bairros(base["bairro"].unique())
geo, quadro = carregar_geojson(tuple(bairros))
risp_de = dados.risp_por_bairro(base)
aisp_de = dados.aisp_por_bairro(base)
geo_aisp = carregar_geojson_aisp(tuple(bairros))
nomes_sem_bairro = dados.poligonos_sem_bairro(geo, bairros)
mes_por_rotulo = {dados.mes_rotulo(p): p for p in dados.MESES}
rotulos_meses = list(mes_por_rotulo)

# o caso explicado fica guardado fora dos controles da barra lateral: com outra aba aberta eles não são
# desenhados, e o Streamlit apaga o estado de controle que não aparece na execução
st.session_state.setdefault("caso_bairro", dados.BAIRRO_INICIAL)
st.session_state.setdefault("caso_mes", dados.mes_rotulo(dados.MES_INICIAL))
st.session_state.setdefault("sel_exploracao", [])
st.session_state.setdefault("agregacao", "bairro")
st.session_state.setdefault("agregacao_contribuicao", "bairro")
aplicar_cliques(bairros)

# a barra lateral só tem o caso explicado, e só com a aba de explicação aberta (o st.tabs, com chave e
# on_change="rerun", põe na sessão o rótulo da aba aberta)
if st.session_state.get("aba", ABAS[0]) == "Explicação":
    with st.sidebar:
        st.markdown("#### Caso explicado")
        st.session_state.setdefault("sel_bairro", st.session_state["caso_bairro"])
        st.session_state.setdefault("sel_mes", st.session_state["caso_mes"])
        st.session_state["caso_bairro"] = st.selectbox("Bairro", bairros, key="sel_bairro")
        st.session_state["caso_mes"] = st.selectbox("Mês-alvo", [dados.mes_rotulo(p) for p in dados.MESES_ALVO],
                                                     key="sel_mes")
bairro = st.session_state["caso_bairro"]
mes_alvo = mes_por_rotulo[st.session_state["caso_mes"]]
linha = dados.linha_da_base(base, bairro, mes_alvo)
mes_t = dados.mes_de_referencia(linha)
assert mes_t == mes_alvo - 1
if st.session_state.get("aba", ABAS[0]) == "Explicação":
    with st.sidebar:
        st.caption("com dados até " + dados.mes_rotulo(mes_t))

rotulo_alvo = dados.mes_rotulo(mes_alvo)
com_modelo = rodada is not None and not DEMONSTRACAO
# os arquivos do teste de 2024 só valem no modo normal com o modelo; a demonstração não muda
metricas = carregar_metricas() if com_modelo else None

st.header("VEX-CVLI. Explicação da estimativa mensal de CVLI por bairro, Salvador", anchor=False)
if metricas is None:
    st.markdown(CABECALHO_COM_MODELO if com_modelo else CABECALHO)

aba_exploracao, aba_explicacao, aba_diagnostico = st.tabs(list(ABAS), key="aba", on_change="rerun")

with aba_exploracao, st.container(key="aba_exploracao"):
    # 2 partes, para as capturas do capítulo: filtros, mapa com o turno e a idade ao lado, e as 3 distribuições da
    # linha de baixo; e a série
    parte_1, parte_2 = st.container(key="exploracao_1"), st.container(key="exploracao_2")
    # os filtros valem só para a exploração. Ficam em linha, sem expansor: com o cabeçalho e as margens dele, a
    # aba passava de 21 cm de altura a 16 cm de largura
    c_periodo, c_natureza, c_turno, c_dias = parte_1.columns([1.2, 1.1, 1.3, 1], gap="medium")
    with c_periodo:
        ini, fim = st.select_slider("Período", options=rotulos_meses, value=(rotulos_meses[0], rotulos_meses[-1]),
                                    key="periodo")
        # seleção da exploração, independente do caso explicado; o clique no mapa também acrescenta ou retira
        # (aplicar_cliques). Fica embaixo do período, na altura livre ao lado das caixas da natureza e do turno
        selecao = st.multiselect("Bairros", bairros, key="sel_exploracao", placeholder="Salvador inteira")
    with c_natureza, st.container(gap="xxsmall"):
        st.markdown("Natureza")
        naturezas = [n for i, (n, rotulo) in enumerate(dados.NATUREZAS.items())
                     if st.checkbox(rotulo, value=True, key="natureza_%d" % i)]
    with c_turno, st.container(gap="xxsmall"):
        st.markdown("Turno")
        turnos = [t for t, (rotulo, faixa) in dados.TURNOS.items()
                  if st.checkbox("%s, %s" % (rotulo.capitalize(), faixa), value=True, key="turno_" + t)]
    with c_dias:
        dias = st.pills("Dia da semana", list(dados.DIAS), format_func=dados.DIAS.get, selection_mode="multi",
                        default=list(dados.DIAS), key="dias")
    periodo = (mes_por_rotulo[ini], mes_por_rotulo[fim])
    todos_os_turnos = dados.filtrar(eventos, periodo, naturezas, None, dias)
    filtrado = todos_os_turnos[todos_os_turnos["turno"].isin(turnos)]
    contagem = dados.contagem_por_bairro(filtrado, bairros)
    # nome da seleção nas legendas e nos títulos: o bairro, com um só; "seleção (N bairros)", com mais
    nome_selecao = (None if not selecao else selecao[0] if len(selecao) == 1
                    else "seleção (%d bairros)" % len(selecao))

    col_mapa, col_direita = parte_1.columns([3, 2], gap="medium")
    with col_mapa, st.container(border=True, key="cartao_mapa"):
        # título e agregação na mesma linha, para o cartão ficar mais baixo
        c_titulo, c_agregacao = st.columns([2.2, 1.6], vertical_alignment="bottom")
        with c_titulo:
            st.markdown("##### Mapa de Salvador")
        with c_agregacao:
            agregacao = st.segmented_control("Agregação", list(ROTULO_AGREGACAO),
                                             format_func=ROTULO_AGREGACAO.get, key="agregacao",
                                             on_change=manter_escolha, args=("agregacao", "bairro"))
        tabela_m = (dados.tabela_aisp(contagem, aisp_de) if agregacao == "aisp"
                    else dados.tabela_mapa(contagem, risp_de, agregacao))
        st.plotly_chart(figura_mapa(tabela_m, "observado", agregacao, selecao, geo, quadro, nomes_sem_bairro,
                                    geo_aisp=geo_aisp),
                        key=chave_do_mapa(), on_select="rerun", selection_mode="points", config=SEM_BARRA)
    with col_direita:
        with st.container(border=True, key="cartao_turno"):
            st.markdown("##### Distribuição por turno")
            dist, totais = dados.distribuicao_turno(todos_os_turnos, selecao)
            st.plotly_chart(figura_turno(dist, totais, nome_selecao, turnos), key="turno", config=SEM_BARRA)
        with st.container(border=True, key="cartao_idade"):
            st.markdown("##### Distribuição por idade da vítima")
            dist_i, totais_i = dados.distribuicao_idade(filtrado, selecao)
            st.plotly_chart(figura_idade(dist_i, totais_i, nome_selecao), key="distribuicao_idade", config=SEM_BARRA)
    col_natureza, col_sexo, col_risp = parte_1.columns([1.2, 1.1, 1], gap="medium")
    with col_natureza, st.container(border=True, key="cartao_natureza"):
        st.markdown("##### Distribuição por natureza")
        dist_n, totais_n = dados.distribuicao_natureza(dados.filtrar(eventos, periodo, None, turnos, dias), selecao)
        st.plotly_chart(figura_natureza(dist_n, totais_n, nome_selecao, naturezas), key="distribuicao_natureza",
                        config=SEM_BARRA)
    with col_sexo, st.container(border=True, key="cartao_sexo"):
        st.markdown("##### Distribuição por gênero")
        dist_s, totais_s = dados.distribuicao_sexo(filtrado, selecao)
        st.plotly_chart(figura_sexo(dist_s, totais_s, nome_selecao), key="distribuicao_sexo", config=SEM_BARRA)
    with col_risp, st.container(border=True, key="cartao_risp"):
        st.markdown("##### Distribuição por RISP")
        dist_r, _ = dados.distribuicao_risp(filtrado, risp_de, rotulos.ORDEM_RISP)
        st.plotly_chart(figura_risp(dist_r, {risp_de[b] for b in selecao}, nome_selecao), key="distribuicao_risp",
                        config=SEM_BARRA)
    with parte_2, st.container(border=True, key="cartao_serie"):
        st.markdown("##### Série mensal " + ("de Salvador" if not selecao else "de " + selecao[0] if len(selecao) == 1
                                             else "da " + nome_selecao))
        st.plotly_chart(figura_serie(dados.serie_mensal(filtrado, selecao, periodo), periodo, None),
                        key="serie", config=SEM_BARRA)

with aba_explicacao:
    if DEMONSTRACAO:
        with st.container(key="linhas_bc"):
            secao_demonstracao()
        with st.container(key="linha_d"):
            secao_agregacao()
    else:
        if rodada is not None:
            secao_modelo(rodada, bairro, mes_alvo, linha, rotulo_alvo)
        else:
            with st.container(key="linhas_bc"):
                secao_reservada(bairro, mes_alvo, linha, rotulo_alvo)
        with st.container(key="explicacao_3"):
            with st.container(border=True, key="cartao_mapa_contribuicao"):
                c_titulo, c_atributo, c_agregacao = st.columns([1.3, 1.5, 1], vertical_alignment="bottom")
                with c_titulo:
                    st.markdown("##### Mapa da contribuição")
                with c_agregacao:
                    agregacao_c = st.segmented_control("Agregação", list(ROTULO_AGREGACAO),
                                                       format_func=ROTULO_AGREGACAO.get, key="agregacao_contribuicao",
                                                       on_change=manter_escolha,
                                                       args=("agregacao_contribuicao", "bairro"))
                # na AISP, com o modelo, os polígonos são os das 16 AISP; sem o modelo, o mapa reservado é o dos bairros
                if agregacao_c == "aisp" and rodada is not None:
                    tabela_c = dados.tabela_aisp(contagem, aisp_de)
                else:
                    tabela_c = dados.tabela_mapa(contagem, risp_de, "bairro" if agregacao_c == "aisp" else agregacao_c)
                atributo_mapa = None
                with c_atributo:
                    if rodada is not None:
                        atributo = st.selectbox("Atributo", [modelo.TOTAL] + rodada.atributos,
                                                format_func=rotulo_atributo, key="atributo")
                        atributo_mapa = rotulo_atributo(atributo)
                    else:
                        st.selectbox("Atributo", [ROTULO_TOTAL] + atributos, disabled=True, key="atributo")
                if rodada is not None:
                    # os filtros da exploração não valem aqui: a média vem do modelo, nas linhas de 2024
                    media = modelo.media_abs(rodada, atributo, agregacao_c)
                    # a coluna da tabela tem o nome da agregação (bairro, aisp ou risp)
                    tabela_c = tabela_c.assign(valor=tabela_c[agregacao_c].map(media))
                st.plotly_chart(figura_mapa(tabela_c, "contribuição", agregacao_c, [bairro], geo, quadro,
                                            nomes_sem_bairro, atributo_mapa, geo_aisp=geo_aisp),
                                key=chave_do_mapa("mapa_contribuicao"), on_select="rerun", selection_mode="points",
                                config=SEM_BARRA)
            # a aba termina no mapa da contribuição: o cartão da série com a estimativa saiu (pedido do orientador
            # em 21/10); a figura_serie e a dados.serie_mensal continuam, porque a exploração as usa

with aba_diagnostico, st.container(key="aba_diagnostico"):
    if metricas is not None:
        secao_diagnostico(rodada, metricas)
    else:
        st.markdown(DIAGNOSTICO)
