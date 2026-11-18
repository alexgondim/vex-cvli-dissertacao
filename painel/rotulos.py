"""Rótulos, cores e letra do painel.

Define a escala dos eixos de contribuição, as cores do SHAP, os anos da mudança entre anos, a cor e a forma das
regiões da dependência e o tamanho mínimo da letra dos gráficos. Guarda os textos de interface dos cartões em
`TEXTOS`: legenda, caixa "Como ler", ajuda das chaves, título do eixo da idade e glossário dos atributos. Não lê nem
grava arquivo. O `app.py` e o `visoes_shap.py` o importam.
"""

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
# tamanho de cada forma com a área visível do círculo, medida em capturas da tela (densidade 2). No
# plotly o tamanho é o diâmetro do círculo e o lado do quadrado, e o triângulo de mesmo tamanho tem pouco mais
# da metade da área do círculo; como a borda come uma faixa de largura quase fixa, a conta geométrica não
# basta. Pontos: área da cor (o contorno é branco), pela mediana dos marcadores da dependência; o quadrado
# anda em degraus de pixel, e com 7,1 a maioria caía no degrau de baixo. Caso: a silhueta inteira, com o
# contorno preto, em marcadores isolados
TAMANHO_PONTO = dict(zip(FORMAS_REGIAO, (8, 7.3, 11.0)))
TAMANHO_CASO = dict(zip(FORMAS_REGIAO, (12, 10.6, 15.4)))
# as regiões são as 3 RISP, na ordem da legenda dos mapas
ORDEM_RISP = ("BTS", "CENTRAL", "ATLÂNTICO")
NOMES_RISP = {"BTS": "Baía de Todos os Santos", "CENTRAL": "Central", "ATLÂNTICO": "Atlântico"}
# letra mínima dos gráficos, em px de tela: numa captura da seção com cerca de 1046 px (1030 mais a folga do
# recorte), reduzida a 16 cm, 14 px dão 6,1 pt
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


# textos de interface: uma tupla por texto, com as colunas do arquivo de textos do painel (tipo, chave, cartão ou
# item, campo e texto), na ordem dele; o painel não lê o arquivo
TEXTOS = (
    ("ajuda", "mapa_observado", "Mapa de Salvador", "O que mostra",
     "O número de registros de CVLI em cada bairro, AISP ou RISP, no período e nos filtros escolhidos."),
    ("ajuda", "mapa_observado", "Mapa de Salvador", "Como ler",
     "Quanto mais escuro o vermelho, mais registros. O contorno preto marca os bairros da seleção. O quadrado "
     "vazio da legenda indica os bairros sem registro na base."),
    ("ajuda", "mapa_observado", "Mapa de Salvador", "Para que serve",
     "Ver onde os registros se concentram na cidade."),
    ("ajuda", "mapa_observado", "Mapa de Salvador", "Cuidados",
     "O mapa mostra contagem de registros, e não taxa por habitante."),
    ("ajuda", "turno", "Distribuição por turno", "O que mostra",
     "Como os registros de CVLI do período se dividem entre os quatro turnos do dia, na seleção e em Salvador."),
    ("ajuda", "turno", "Distribuição por turno", "Como ler",
     "A barra azul é a seleção e a cinza é Salvador. A altura é o percentual dos registros, sempre calculado sobre "
     "os 4 turnos. Barra esmaecida é turno desmarcado no filtro."),
    ("ajuda", "turno", "Distribuição por turno", "Para que serve",
     "Comparar o horário dos registros de um bairro com o da cidade."),
    ("ajuda", "turno", "Distribuição por turno", "Cuidados",
     "Com poucos registros na seleção, os percentuais variam muito. O número de registros aparece na legenda."),
    ("ajuda", "idade", "Distribuição por idade da vítima", "O que mostra",
     "Como os registros de CVLI com idade informada se dividem por faixa de idade da vítima, na seleção e em "
     "Salvador."),
    ("ajuda", "idade", "Distribuição por idade da vítima", "Como ler",
     "A barra azul é a seleção e a cinza, atrás dela, é Salvador. A altura é o percentual dos registros com idade "
     "informada. O número sobre as barras é o percentual da seleção; sem seleção, o de Salvador."),
    ("ajuda", "idade", "Distribuição por idade da vítima", "Para que serve",
     "Comparar a idade das vítimas de um bairro com a da cidade."),
    ("ajuda", "idade", "Distribuição por idade da vítima", "Cuidados",
     "Registros sem idade informada ficam fora do cálculo. Com poucos registros na seleção, os percentuais variam "
     "muito."),
    ("ajuda", "natureza", "Distribuição por natureza", "O que mostra",
     "Como os registros de CVLI do período se dividem entre as quatro naturezas, na seleção e em Salvador."),
    ("ajuda", "natureza", "Distribuição por natureza", "Como ler",
     "A barra azul é a seleção e a cinza é Salvador. O comprimento é o percentual dos registros, sempre calculado "
     "sobre as 4 naturezas. Barra esmaecida é natureza desmarcada no filtro."),
    ("ajuda", "natureza", "Distribuição por natureza", "Para que serve",
     "Comparar a composição dos registros de um bairro com a da cidade."),
    ("ajuda", "natureza", "Distribuição por natureza", "Cuidados",
     "Com poucos registros na seleção, os percentuais variam muito."),
    ("ajuda", "genero", "Distribuição por gênero", "O que mostra",
     "A proporção de vítimas do gênero masculino e do feminino nos registros de CVLI do período."),
    ("ajuda", "genero", "Distribuição por gênero", "Como ler",
     "O anel de fora é a seleção e o de dentro é Salvador. Sem seleção, aparece um anel só, o da cidade."),
    ("ajuda", "genero", "Distribuição por gênero", "Para que serve",
     "Comparar a composição por gênero de um bairro com a da cidade."),
    ("ajuda", "genero", "Distribuição por gênero", "Cuidados",
     "Com poucos registros na seleção, os percentuais variam muito."),
    ("ajuda", "risp", "Distribuição por RISP", "O que mostra",
     "Como os registros de CVLI de Salvador no período se dividem entre as três RISP."),
    ("ajuda", "risp", "Distribuição por RISP", "Como ler",
     "Cada fatia é uma RISP. Ficam em azul as RISP que têm bairro na seleção, e as outras ficam em cinza. Sem "
     "seleção, todas ficam em cinza. O gráfico é sempre da cidade inteira."),
    ("ajuda", "risp", "Distribuição por RISP", "Para que serve",
     "Situar a região da seleção no total da cidade."),
    ("ajuda", "risp", "Distribuição por RISP", "Cuidados",
     "As RISP reúnem números diferentes de bairros, o que influi no tamanho das fatias."),
    ("ajuda", "serie", "Série mensal", "O que mostra",
     "O número de registros de CVLI em cada mês, em Salvador, em um bairro ou na seleção."),
    ("ajuda", "serie", "Série mensal", "Como ler",
     "A linha liga as contagens dos meses. As faixas de fundo marcam a pandemia, em 2020 e 2021, e o ano de teste "
     "do modelo, 2024."),
    ("ajuda", "serie", "Série mensal", "Para que serve",
     "Ver como os registros variaram ao longo do tempo."),
    ("ajuda", "serie", "Série mensal", "Cuidados",
     "Em um bairro, a contagem mensal é baixa e muitos meses têm zero. Uma diferença de um mês para outro pode ser "
     "só oscilação."),
    ("ajuda", "cascata", "Explicação local", "O que mostra",
     "Como o modelo chegou à estimativa de um bairro em um mês. A conta parte de um valor de base, igual para "
     "todos os casos, e soma a contribuição de cada atributo."),
    ("ajuda", "cascata", "Explicação local", "Como ler",
     "Barra vermelha aumenta a estimativa e barra azul reduz. Quanto mais longa a barra, maior a contribuição. "
     "Cada linha traz o atributo e o valor dele naquele caso. Com a chave dos 30 atributos desligada, a última "
     "linha soma as contribuições dos demais. No alto do cartão ficam o mês estimado, a estimativa e o observado."),
    ("ajuda", "cascata", "Explicação local", "Para que serve",
     "Ver quais atributos levaram a estimativa daquele caso para cima ou para baixo."),
    ("ajuda", "cascata", "Explicação local", "Cuidados",
     "As contribuições estão no logaritmo da contagem esperada, e não em número de registros. Elas mostram o que o "
     "modelo usou para estimar, e não a causa dos crimes. A estimativa é um valor esperado e pode ficar longe do "
     "observado em um mês isolado."),
    ("ajuda", "enxame", "Resumo global", "O que mostra",
     "A contribuição de cada atributo em todas as linhas do teste de 2024. Cada ponto é um bairro em um mês."),
    ("ajuda", "enxame", "Resumo global", "Como ler",
     "Ponto à direita do zero aumenta a estimativa; à esquerda, reduz. A cor indica o valor do atributo, do azul, "
     "baixo, ao vermelho, alto. Ponto cinza é de atributo sem valor alto ou baixo (bairro, AISP e RISP), e "
     "cinza-escuro é valor ausente. Os atributos aparecem em ordem de peso, com o de maior peso no alto. O "
     "contorno preto marca o caso selecionado."),
    ("ajuda", "enxame", "Resumo global", "Para que serve",
     "Ver quais atributos mais pesam nas estimativas e em que sentido."),
    ("ajuda", "enxame", "Resumo global", "Cuidados",
     "A escala é o logaritmo da contagem esperada. Contribuição não é causa. A cor é relativa a cada atributo: o "
     "mesmo vermelho em dois atributos não indica o mesmo valor."),
    ("ajuda", "dependencia", "Dependência (as quatro abas)", "O que mostra",
     "Como a contribuição de um atributo varia com o valor dele, nas linhas do teste de 2024. As abas trocam o "
     "atributo."),
    ("ajuda", "dependencia", "Dependência (as quatro abas)", "Como ler",
     "Cada ponto é um bairro em um mês. O eixo horizontal traz o valor do atributo e o vertical, a contribuição. "
     "Ponto acima do zero aumenta a estimativa; abaixo, reduz. A cor e a forma indicam a RISP do bairro. O "
     "contorno preto marca o caso selecionado."),
    ("ajuda", "dependencia", "Dependência (as quatro abas)", "Para que serve",
     "Ver se a contribuição cresce ou diminui com o valor do atributo e se isso muda de uma RISP para outra."),
    ("ajuda", "dependencia", "Dependência (as quatro abas)", "Cuidados",
     "Nas abas de turno, as linhas sem registro nos doze meses anteriores não aparecem. O gráfico mostra uma "
     "associação aprendida pelo modelo, e não uma causa."),
    ("ajuda", "anos", "Mudança entre anos", "O que mostra",
     "A distribuição das contribuições de cada atributo em três anos: 2019 e 2021, do treino, e 2024, do teste."),
    ("ajuda", "anos", "Mudança entre anos", "Como ler",
     "Cada atributo tem três caixas, uma por ano. A caixa cobre a metade central dos valores, e o traço dentro "
     "dela é a mediana. Os pontos soltos são valores afastados dos demais. Valor à direita do zero aumenta a "
     "estimativa; à esquerda, reduz."),
    ("ajuda", "anos", "Mudança entre anos", "Para que serve",
     "Ver se o peso de um atributo nas estimativas mudou de um ano para outro."),
    ("ajuda", "anos", "Mudança entre anos", "Cuidados",
     "O modelo é o mesmo nos três anos. Uma diferença entre as caixas vem dos valores dos atributos em cada ano, e "
     "não de mudança no modelo."),
    ("ajuda", "mapa_contribuicao", "Mapa da contribuição", "O que mostra",
     "Onde o atributo escolhido mais pesa nas estimativas de 2024. A cor é a média do valor absoluto da "
     "contribuição dele, por bairro, AISP ou RISP. Na opção de todos os atributos, a cor é a média da soma dos "
     "valores absolutos das 30 contribuições."),
    ("ajuda", "mapa_contribuicao", "Mapa da contribuição", "Como ler",
     "Quanto mais escuro o laranja, maior o peso naquele lugar. O mapa mostra o tamanho da contribuição, sem dizer "
     "se ela aumenta ou reduz a estimativa. O contorno preto marca o bairro do caso selecionado."),
    ("ajuda", "mapa_contribuicao", "Mapa da contribuição", "Para que serve",
     "Localizar as partes da cidade em que o atributo mais influencia as estimativas."),
    ("ajuda", "mapa_contribuicao", "Mapa da contribuição", "Cuidados",
     "Peso alto não quer dizer mais registros. Quer dizer que o atributo escolhido, ou o conjunto dos 30, moveu "
     "mais a estimativa, para cima ou para baixo. As cores usam sempre as linhas de 2024 e não mudam com o mês do "
     "caso nem com os filtros da exploração."),
    ("ajuda", "corte_fixo", "Métricas por corte fixo", "O que mostra",
     "O erro do modelo e de quatro métodos de referência no teste de 2024, em três medidas: MAE, RMSE e deviance "
     "de Poisson."),
    ("ajuda", "corte_fixo", "Métricas por corte fixo", "Como ler",
     "Barra menor é erro menor. A barra azul é o modelo, e as cinzas são as referências."),
    ("ajuda", "corte_fixo", "Métricas por corte fixo", "Para que serve",
     "Comparar o erro do modelo com o de métodos mais simples."),
    ("ajuda", "corte_fixo", "Métricas por corte fixo", "Cuidados",
     "As três medidas podem ordenar os métodos de formas diferentes. O resultado vale para 2024, o único ano de "
     "teste."),
    ("ajuda", "janela", "Métricas por ano na janela expansiva", "O que mostra",
     "O erro de cada método em 2022, 2023 e 2024, com o modelo ajustado de novo a cada mês."),
    ("ajuda", "janela", "Métricas por ano na janela expansiva", "Como ler",
     "Cada linha é um método, e a azul é o modelo. Ponto mais baixo é erro menor."),
    ("ajuda", "janela", "Métricas por ano na janela expansiva", "Para que serve",
     "Ver se a posição do modelo em relação às referências se mantém de um ano para outro."),
    ("ajuda", "janela", "Métricas por ano na janela expansiva", "Cuidados",
     "Só 2024 é ano de teste. Em 2022 e 2023, a avaliação se sobrepõe à escolha da configuração do modelo."),
    ("ajuda", "mensal", "Métricas por meses na janela expansiva", "O que mostra",
     "O erro de cada método em cada um dos 36 meses de janeiro de 2022 a dezembro de 2024, na medida escolhida."),
    ("ajuda", "mensal", "Métricas por meses na janela expansiva", "Como ler",
     "Cada linha é um método, e a azul é o modelo. Ponto mais baixo é erro menor."),
    ("ajuda", "mensal", "Métricas por meses na janela expansiva", "Para que serve",
     "Localizar os meses em que o erro se concentra."),
    ("ajuda", "mensal", "Métricas por meses na janela expansiva", "Cuidados",
     "O erro de um mês isolado oscila. Só os meses de 2024 são de teste."),
    ("ajuda", "tabela", "Erro por forma de validação, ano e método", "O que mostra",
     "Os valores dos dois primeiros gráficos da aba, por forma de validação, ano e método. Os valores mensais do "
     "terceiro gráfico não entram na tabela."),
    ("ajuda", "tabela", "Erro por forma de validação, ano e método", "Como ler",
     "Cada linha é um método em um ano. Valor menor é erro menor."),
    ("ajuda", "tabela", "Erro por forma de validação, ano e método", "Para que serve",
     "Consultar, com 3 casas decimais, os valores dos gráficos de corte fixo e de janela por ano."),
    ("ajuda", "tabela", "Erro por forma de validação, ano e método", "Cuidados",
     "Comparar apenas valores da mesma medida e da mesma forma de validação."),
    ("legenda", "mapa_observado", "Mapa de Salvador", "legenda sob o título",
     "Registros de CVLI por {nível}, de {inicio} a {fim}, nos filtros escolhidos. Contorno preto: bairros da "
     "seleção."),
    ("legenda", "turno", "Distribuição por turno", "legenda sob o título",
     "Registros de CVLI por turno, de {inicio} a {fim}. Barras esmaecidas: turnos desmarcados no filtro."),
    ("legenda", "idade", "Distribuição por idade da vítima", "legenda sob o título",
     "Registros de CVLI com idade informada, por faixa de idade, de {inicio} a {fim}."),
    ("legenda", "natureza", "Distribuição por natureza", "legenda sob o título",
     "Registros de CVLI por natureza, de {inicio} a {fim}. Barras esmaecidas: naturezas desmarcadas no filtro."),
    ("legenda", "genero", "Distribuição por gênero", "legenda sob o título",
     "Registros de CVLI por gênero da vítima, de {inicio} a {fim}."),
    ("legenda", "risp", "Distribuição por RISP", "legenda sob o título",
     "Registros de CVLI de Salvador por RISP, de {inicio} a {fim}."),
    ("legenda", "serie", "Série mensal", "legenda sob o título",
     "Registros de CVLI por mês."),
    ("legenda", "cascata", "Explicação local", "legenda sob o título",
     "Mês estimado: {mês} · Estimativa: {estimativa} · Observado: {observado} (registros de CVLI no mês)"),
    ("legenda", "enxame", "Resumo global", "legenda sob o título",
     "Contribuição de cada atributo para a estimativa. Contribuição positiva aumenta a estimativa; negativa reduz."),
    ("legenda", "dependencia", "Dependência (as quatro abas)", "legenda sob o título",
     "Contribuição do atributo da aba aberta conforme o valor dele. Contribuição positiva aumenta a estimativa; "
     "negativa reduz."),
    ("legenda", "anos", "Mudança entre anos", "legenda sob o título",
     "Distribuição das contribuições de cada atributo em cada ano. Contribuição positiva aumenta a estimativa; "
     "negativa reduz."),
    ("legenda", "mapa_contribuicao", "Mapa da contribuição", "legenda sob o título",
     "Média do valor absoluto da contribuição do atributo escolhido, ou dos 30 somados, no teste de 2024. Contorno "
     "preto: bairro do caso selecionado."),
    ("legenda", "corte_fixo", "Métricas por corte fixo", "legenda sob o título",
     "Teste de 2024. MAE e RMSE em registros por bairro e mês."),
    ("legenda", "janela", "Métricas por ano na janela expansiva", "legenda sob o título",
     "MAE e RMSE em registros por bairro e mês. Só 2024 é ano de teste."),
    ("legenda", "mensal", "Métricas por meses na janela expansiva", "legenda sob o título",
     "MAE e RMSE em registros por bairro e mês. Só os meses de 2024 são de teste."),
    ("legenda", "tabela", "Erro por forma de validação, ano e método", "legenda sob o título",
     "MAE e RMSE em registros por bairro e mês."),
    ("ajuda_da_chave", "chave_30", "ver os 30 atributos", "texto de ajuda",
     "Mostra os 30 atributos na explicação local, no resumo global e na mudança entre anos. Desligada, o resumo "
     "global e a mudança entre anos mostram os 10 de maior peso, e a explicação local mostra os 9 de maior "
     "contribuição no caso e soma os demais na última linha."),
    ("ajuda_da_chave", "chave_escala", "bairro em escala própria", "texto de ajuda",
     "Dá ao atributo bairro um eixo só dele no resumo global e na mudança entre anos, para que os demais atributos "
     "apareçam em escala maior. Muda só o desenho: as contribuições são as mesmas."),
    ("eixo", "eixo_idade", "Distribuição por idade da vítima, eixo horizontal", "título do eixo",
     "idade da vítima (anos)"),
    ("glossario", "glossario", "cvli_lag1 a cvli_lag12", "definição",
     "Contagem de CVLI do bairro em cada um dos doze meses anteriores ao mês estimado. O cvli_lag1 é o mês mais "
     "recente."),
    ("glossario", "glossario", "cvli_ma3, cvli_ma6, cvli_ma12", "definição",
     "Média das 3, das 6 e das 12 contagens mensais mais recentes do bairro."),
    ("glossario", "glossario", "cvli_trend", "definição",
     "Diferença entre as contagens dos dois meses mais recentes (cvli_lag1 menos cvli_lag2)."),
    ("glossario", "glossario", "month_sin_alvo, month_cos_alvo", "definição",
     "Mês do ano que está sendo estimado, escrito em seno e cosseno para que dezembro e janeiro fiquem vizinhos."),
    ("glossario", "glossario", "year_norm_alvo", "definição",
     "Ano que está sendo estimado, de 0 (2018) a 1 (2024)."),
    ("glossario", "glossario", "time_index", "definição",
     "Posição, na série, do mês anterior ao estimado, contada a partir de janeiro de 2018, que vale 0."),
    ("glossario", "glossario", "prop_madrugada_hist12, prop_tarde_hist12, prop_noite_hist12", "definição",
     "Proporção dos registros do bairro nos doze meses anteriores que ocorreram em cada turno."),
    ("glossario", "glossario", "prop_fds_hist12", "definição",
     "Proporção dos registros do bairro nos doze meses anteriores que ocorreram no sábado ou no domingo."),
    ("glossario", "glossario", "bairro, aisp, risp", "definição",
     "O bairro da linha e a AISP e a RISP a que ele pertence."),
    ("glossario", "glossario", "prop_masc_hist12", "definição",
     "Proporção de vítimas do gênero masculino nos doze meses anteriores."),
    ("glossario", "glossario", "idade_media_hist12", "definição",
     "Média das idades informadas das vítimas nos doze meses anteriores."),
    ("glossario", "glossario", "n_vitimas_hist12", "definição",
     "Número de vítimas de CVLI no bairro nos doze meses anteriores."),
    ("rotulo", "rotulo_ajuda", "controle de ajuda de cada cartão", "rótulo",
     "Como ler"),
    ("rotulo", "rotulo_glossario", "caixa do glossário, na aba de explicação", "rótulo",
     "O que é cada atributo"),
)


def _ajuda(textos):
    """{chave do cartão: [(campo, texto)]}, na ordem do arquivo."""
    saida = {}
    for tipo, chave, _, campo, texto in textos:
        if tipo == "ajuda":
            saida.setdefault(chave, []).append((campo, texto))
    return saida


AJUDA = _ajuda(TEXTOS)
LEGENDA = {chave: texto for tipo, chave, _, _, texto in TEXTOS if tipo == "legenda"}
AJUDA_DA_CHAVE = {chave: texto for tipo, chave, _, _, texto in TEXTOS if tipo == "ajuda_da_chave"}
EIXO = {chave: texto for tipo, chave, _, _, texto in TEXTOS if tipo == "eixo"}
ROTULO = {chave: texto for tipo, chave, _, _, texto in TEXTOS if tipo == "rotulo"}
# (atributo ou atributos, definição), na ordem do arquivo, e o cabeçalho das 2 colunas da tabela do glossário
GLOSSARIO = [(item, texto) for tipo, _, item, _, texto in TEXTOS if tipo == "glossario"]
COLUNAS_GLOSSARIO = ("atributo", "definição")


def legenda(chave, valores=None):
    """Legenda do cartão com cada campo entre chaves ({nível}, {inicio}, {fim}, {mês}, {estimativa}, {observado},
    {atributo por extenso}) trocado pelo valor dado em `valores`; campo que sobra na legenda é erro."""
    texto = LEGENDA[chave]
    for campo, valor in (valores or {}).items():
        texto = texto.replace("{%s}" % campo, valor)
    assert "{" not in texto, texto
    return texto
