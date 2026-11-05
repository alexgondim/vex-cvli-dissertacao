"""Figuras do mapa para a dissertação, período completo e sem filtro, com 1600 px de largura. O
recorte é só a área do mapa, com a legenda de cor e a atribuição, sem o título do cartão e sem os
botões de modo e agregação: na figura das RISP eles marcariam "observado + RISP", que no app mostra
totais, e a figura mostra pertencimento.

- mapa_bairros_observado.png: o mapa do app no modo observado, agregação por bairro;
- mapa_risp.png: pertencimento de cada bairro à sua RISP, em 3 cores, com os limites e os números
  das 16 AISP por cima e legenda só com os nomes. O app não tem essa visão (a agregação por RISP
  dele mostra totais); aqui o mapa é recolorido na página da captura.

Com o app rodando (./venv/Scripts/python.exe -m streamlit run painel_mensal/app.py):
python painel_mensal/figuras_mapa.py http://localhost:8501 --pasta DESTINO [bairros] [risp]
Sem o nome de uma figura, gera as duas. O destino é obrigatório, para uma execução nova não
sobrescrever figuras já aprovadas.
O aplicativo sempre tem um bairro selecionado (a lista não tem opção vazia) e desenha o contorno
dele num traço à parte; aqui esse traço é escondido na página da captura, e as figuras saem sem
bairro destacado. O app não muda.

Cada figura entra no capítulo com 7,8 cm, e o mapa tem cerca de 636 px de tela: para passar de 6 pt,
a letra precisa de 17,3 px. Na página da captura, a letra da figura (números das AISP, legendas, barra
de cor e crédito do mapa de fundo) sobe para LETRA_FIGURA px. O mapa de fundo do app não tem nomes.

O quadro também muda na página da captura. No cartão da exploração, o app enquadra os 166 bairros, e as
ilhas dos Frades e de Bom Jesus dos Passos, sem registro, ficam de fora; a legenda da figura diz que os 4
polígonos sem registro aparecem, então aqui o quadro cabe os 170 na largura do mapa. As legendas ficam
onde estavam nas figuras do capítulo de 30/09, todas à esquerda: a barra de cor e a legenda das RISP sobre a
baía, com a base a 20% da altura, e a nota dos polígonos sem cor no canto de baixo.
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import shapely
from PIL import Image
from playwright.sync_api import sync_playwright
from shapely.geometry import Polygon, shape
from shapely.ops import polylabel

import dados
import rotulos
from capturas import ARGS_CHROME, JS_TRACOS, esperar

LARGURA_FIGURA = 1600
VIEWPORT = {"width": 1600, "height": 1300}
# no mapa de cerca de 636 px de tela, 19 px dão 6,6 pt a 7,8 cm
LETRA_FIGURA = 19
# px de tela entre os polígonos e as bordas laterais do mapa
FOLGA_LATERAL = 8
FIGURAS = {"bairros": "mapa_bairros_observado.png", "risp": "mapa_risp.png"}
NOMES_RISP = {"BTS": "Baía de Todos os Santos", "CENTRAL": "Central", "ATLÂNTICO": "Atlântico"}
# único trio claro, entre ColorBrewer Pastel1, Pastel2, Set2, Set3 e Paired e os tons claros do Tableau,
# que separa as 3 regiões em visão normal (17,1), nas simulações de daltonismo (8,0 no pior caso) e do
# fundo cinza-claro (11,9), em distância OKLab x 100, com opacidade 0,8
CORES_RISP = {"Baía de Todos os Santos": "#80b1d3", "Central": "#a6d854", "Atlântico": "#fb8072"}
COR_AISP = "#303030"
LARGURA_AISP = 2.5  # o contorno branco dos bairros tem 0,6
# os bairros vizinhos do GeoJSON nem sempre encostam; 0,00005 grau (cerca de 5 m) de ida e volta
# fecha essas frestas, que virariam linhas falsas dentro das AISP
FECHO_FRESTAS = 5e-5
FONTE_NUMEROS = "Open Sans Bold"  # uma das fontes do próprio estilo Positron
TAMANHO_NUMEROS = 20
# contorno branco fino nos números: o 14 e o 15 caem em cima de limites de AISP
HALO_NUMEROS = 1.5
AISP_POR_RISP = {"BTS": {1, 2, 3, 4, 5, 8}, "ATLÂNTICO": {6, 7, 9, 12, 14, 15, 16}, "CENTRAL": {10, 11, 13}}

GRAFICO = """Array.from(document.querySelectorAll('.js-plotly-plot'))
    .find(d => d._fullData && d._fullData.length && d._fullData[0].type === 'choroplethmap')"""
JS_MAPA = """() => {
  const gd = %s;
  const escala = gd._fullLayout.coloraxis.colorscale;
  return {fundo: gd._fullLayout.map.style, escala: [escala[0], escala[escala.length - 1]],
          barras_de_cor: gd.querySelectorAll('.colorbar').length};
}""" % GRAFICO
JS_ATRIBUICAO = """() => {
  const a = document.querySelector('.maplibregl-ctrl-attrib');
  const texto = document.querySelector('.maplibregl-ctrl-attrib-inner');
  return {visivel: getComputedStyle(texto).display !== 'none', texto: a.innerText.trim()};
}"""
# o contorno do bairro selecionado é o traço de polígono com um só local; escondê-lo só na página tira
# o destaque sem mudar o que o app manda desenhar
JS_SEM_DESTAQUE = """() => {
  const gd = %s;
  const i = gd._fullData.findIndex(t => t.type === 'choroplethmap' && (t.locations || []).length === 1);
  const locais = Array.from(gd._fullData[i].locations || []);
  return Plotly.restyle(gd, {visible: false}, [i]).then(() => ({traco: i, locais: locais, visivel: gd._fullData[i].visible}));
}""" % GRAFICO
# troca a cor por total da RISP pela cor da RISP: código 0, 1 ou 2 por polígono e a escala de cor do
# mapa em 3 degraus, sem barra; a legenda dos nomes entra como anotação
JS_PERTENCIMENTO = """(a) => {
  const gd = %s;
  const locais = Array.from(gd._fullData[0].locations);
  const z = locais.map(l => a.codigo[l]);
  const n = a.cores.length;
  const escala = [];
  a.cores.forEach((c, i) => { escala.push([i / n, c]); escala.push([(i + 1) / n, c]); });
  return Plotly.restyle(gd, {z: [z]}, [0])
    .then(() => Plotly.relayout(gd, {'coloraxis.colorscale': escala, 'coloraxis.cmin': -0.5, 'coloraxis.cmax': n - 0.5,
                                     'coloraxis.showscale': false,
                                     annotations: (gd.layout.annotations || []).concat([a.legenda])}))
    .then(() => ({sem_codigo: locais.filter(l => a.codigo[l] === undefined),
                  por_regiao: a.cores.map((c, k) => z.filter(v => v === k).length),
                  escala: gd._fullLayout.coloraxis.colorscale}));
}""" % GRAFICO
# limites e números das AISP por cima das cores: below '' como nos polígonos do app, que ficam acima de todas
# as camadas do mapa de fundo; sem isso eles entram logo acima do fundo, embaixo dos polígonos. O MapLibre
# esconde rótulo que encosta em outro, então os números são forçados a aparecer
JS_AISP = """(a) => {
  const gd = %s;
  return Plotly.addTraces(gd, [
      {type: 'scattermap', mode: 'lines', lon: a.lon, lat: a.lat, line: {color: a.cor, width: a.largura},
       below: '', hoverinfo: 'skip', showlegend: false},
      {type: 'scattermap', mode: 'text', lon: a.num_lon, lat: a.num_lat, text: a.num_txt,
       textfont: {family: a.fonte, size: a.tamanho, color: a.cor}, below: '', hoverinfo: 'skip', showlegend: false}])
    .then(() => {
      const map = gd._fullLayout.map._subplot.map;
      const uid = gd._fullData[gd._fullData.length - 1].uid;
      const ids = map.getStyle().layers.map(l => l.id).filter(id => id.includes(uid) && map.getLayer(id).type === 'symbol');
      ids.forEach(id => {
        map.setLayoutProperty(id, 'text-allow-overlap', true);
        map.setPaintProperty(id, 'text-halo-color', '#ffffff');
        map.setPaintProperty(id, 'text-halo-width', a.halo);
      });
      return ids;
    });
}""" % GRAFICO
# quadro dos 170 polígonos, barra de cor sobre a baía e nota dos polígonos sem cor no canto de baixo à
# esquerda; devolve o que o mapa ficou mostrando
JS_QUADRO = """(a) => {
  const gd = %s;
  const nota = (gd.layout.annotations || []).findIndex(n => n.text.includes('sem registro na base'));
  const mudar = {'map.center': a.centro, 'map.zoom': a.zoom, 'coloraxis.colorbar.x': 0.01,
                 'coloraxis.colorbar.xanchor': 'left', 'coloraxis.colorbar.y': 0.2, 'coloraxis.colorbar.yanchor': 'bottom'};
  const p = 'annotations[' + nota + '].';
  Object.assign(mudar, {[p + 'x']: 0.01, [p + 'xanchor']: 'left', [p + 'y']: 0.01, [p + 'yanchor']: 'bottom'});
  return Plotly.relayout(gd, mudar)
    .then(() => { const m = gd._fullLayout.map._subplot.map;
                  return {zoom: m.getZoom(), centro: m.getCenter(), estilo: gd._fullLayout.map.style, nota: nota}; });
}""" % GRAFICO
# letra da figura: anotações, barra de cor e crédito do mapa de fundo (este por CSS, porque é do MapLibre)
JS_LETRA = """(a) => {
  const gd = %s;
  const estilo = document.createElement('style');
  estilo.textContent = '.maplibregl-ctrl-attrib, .maplibregl-ctrl-attrib a { font-size: ' + a.tamanho + 'px !important; }';
  document.head.appendChild(estilo);
  const mudar = {};
  (gd.layout.annotations || []).forEach((n, i) => { mudar['annotations[' + i + '].font.size'] = a.tamanho; });
  if (gd._fullLayout.coloraxis && gd._fullLayout.coloraxis.showscale) {
    mudar['coloraxis.colorbar.tickfont.size'] = a.tamanho;
    mudar['coloraxis.colorbar.title.font.size'] = a.tamanho;
  }
  return Plotly.relayout(gd, mudar).then(() => Object.keys(mudar));
}""" % GRAFICO
# o Streamlit refaz o gráfico, com uma instância nova do mapa, a cada execução do app (o clique em RISP
# dispara uma, às vezes duas), e a instância nova perde os ajustes feitos na página; cada instância ganha
# uma marca para os ajustes só começarem com o mapa parado e a figura só ser gravada no mesmo mapa
JS_MARCA_DO_MAPA = """() => {
  const map = (%s)._fullLayout.map._subplot.map;
  if (!map.__marca) map.__marca = Math.random().toString(36).slice(2);
  return map.__marca;
}""" % GRAFICO
JS_NUMEROS_NA_TELA = """(ids) => {
  const gd = %s;
  const map = gd._fullLayout.map._subplot.map;
  return map.queryRenderedFeatures({layers: ids}).map(f => JSON.stringify(f.properties));
}""" % GRAFICO


def base_territorial():
    """Bairro, AISP e RISP de cada bairro. A coluna aisp da base bairro-mês é a que o gerador monta com
    a tabela da Portaria SSP 435/2020 (Anexo I)."""
    t = pd.read_csv(dados.ARQUIVOS["base_bairro_mes"], usecols=["bairro", "aisp", "risp"]).drop_duplicates()
    assert t["bairro"].is_unique and len(t) == 166 and t["aisp"].nunique() == 16, t.describe()
    obtido = {r: set(int(a) for a in g["aisp"]) for r, g in t.groupby("risp")}
    assert obtido == AISP_POR_RISP, obtido
    return t


def geometria_aisp(territorio):
    """Limites (linhas, com None entre os anéis) e ponto do número de cada AISP."""
    formas = {f["properties"][dados.CHAVE_POLIGONO]: shape(f["geometry"]) for f in dados.ler_geojson()["features"]}
    lon, lat, numeros, resumo = [], [], [], {}
    for aisp, g in territorio.groupby("aisp"):
        uniao = shapely.unary_union([formas[dados.poligono(b)].buffer(0) for b in g["bairro"]])
        uniao = uniao.buffer(FECHO_FRESTAS).buffer(-FECHO_FRESTAS)
        outras = [formas[dados.poligono(b)] for b in territorio.loc[territorio["aisp"] != aisp, "bairro"]]
        partes, descartados = [], 0
        for p in (uniao.geoms if uniao.geom_type == "MultiPolygon" else [uniao]):
            # buraco só é limite se outro bairro o ocupa; buraco vazio é fresta do GeoJSON
            buracos = [h for h in p.interiors if any(o.intersection(Polygon(h)).area > 0 for o in outras)]
            descartados += len(p.interiors) - len(buracos)
            partes.append(Polygon(p.exterior, buracos))
        for p in partes:
            for anel in [p.exterior, *p.interiors]:
                x, y = anel.xy
                lon += list(x) + [None]
                lat += list(y) + [None]
        maior = max(partes, key=lambda p: p.area)
        ponto = polylabel(maior, tolerance=1e-5)
        assert maior.contains(ponto), aisp
        numeros.append((ponto.x, ponto.y, str(int(aisp))))
        resumo[int(aisp)] = {"partes": len(partes), "buracos": sum(len(p.interiors) for p in partes),
                             "buracos vazios descartados": descartados}
    return lon, lat, numeros, resumo


def mapa_parado(page, pausa=3.0, tentativas=10):
    """Espera o mesmo mapa ficar na tela por `pausa` segundos e devolve a marca dele."""
    marca = page.evaluate(JS_MARCA_DO_MAPA)
    for _ in range(tentativas):
        time.sleep(pausa)
        agora = page.evaluate(JS_MARCA_DO_MAPA)
        if agora == marca:
            return marca
        marca = agora
    raise RuntimeError("o Streamlit não parou de refazer o mapa")


def esconder_destaque(page):
    """Esconde o traço de contorno e confere que era o do bairro selecionado."""
    resultado = page.evaluate(JS_SEM_DESTAQUE)
    assert resultado["locais"] == [dados.poligono(dados.BAIRRO_INICIAL)] and resultado["visivel"] is False, resultado
    time.sleep(1.5)  # o mapa redesenha as camadas
    return resultado


def colorir_por_risp(page, territorio):
    nomes = list(CORES_RISP)
    codigo = {dados.poligono(b): nomes.index(NOMES_RISP[r]) for b, r in zip(territorio["bairro"], territorio["risp"])}
    # o quadrado e o traço acompanham a letra da legenda, na proporção de antes (17 e 15 para 14)
    linhas = ["<span style='color:%s;font-size:%dpx'>■</span> %s" % (cor, round(LETRA_FIGURA * 17 / 14), nome)
              for nome, cor in CORES_RISP.items()]
    linhas.append("<span style='color:%s;font-size:%dpx'>━</span> limite de AISP"
                  % (COR_AISP, round(LETRA_FIGURA * 15 / 14)))
    legenda = dict(text="<b>RISP</b><br>" + "<br>".join(linhas),
                   x=0.01, y=0.2, xref="paper", yref="paper", xanchor="left", yanchor="bottom", align="left",
                   showarrow=False, bgcolor="rgba(255,255,255,0.9)", borderpad=6,
                   font=dict(size=LETRA_FIGURA, color=rotulos.COR_TEXTO))
    resultado = page.evaluate(JS_PERTENCIMENTO, {"codigo": codigo, "cores": list(CORES_RISP.values()), "legenda": legenda})
    assert not resultado["sem_codigo"] and sum(resultado["por_regiao"]) == 166, resultado
    time.sleep(1.5)
    return dict(zip(nomes, resultado["por_regiao"]))


def enquadrar(page, geo):
    """Quadro dos 170 polígonos na largura do mapa, com FOLGA_LATERAL de cada lado, e barra de cor sobre a
    baía; espera os blocos do mapa de fundo da área nova."""
    caixa = page.locator(".st-key-mapa").bounding_box()
    centro, zoom = dados.enquadramento(geo, [f["properties"][dados.CHAVE_POLIGONO] for f in geo["features"]],
                                       caixa["height"], largura=caixa["width"] - 2 * FOLGA_LATERAL)
    resultado = page.evaluate(JS_QUADRO, {"centro": centro, "zoom": zoom})
    assert resultado["nota"] >= 0, resultado
    time.sleep(4.0)
    return resultado


def desenhar_aisp(page, territorio):
    """Desenha limites e números das AISP; devolve as camadas dos números e o resumo por AISP."""
    lon, lat, numeros, resumo = geometria_aisp(territorio)
    ids = page.evaluate(JS_AISP, {"lon": lon, "lat": lat, "cor": COR_AISP, "largura": LARGURA_AISP,
                                  "num_lon": [n[0] for n in numeros], "num_lat": [n[1] for n in numeros],
                                  "num_txt": [n[2] for n in numeros], "fonte": FONTE_NUMEROS,
                                  "tamanho": TAMANHO_NUMEROS, "halo": HALO_NUMEROS})
    time.sleep(2.0)  # as fontes dos números
    return ids, resumo


def mostrar_atribuicao(page):
    """O MapLibre abre a atribuição do mapa de fundo ao carregar e a recolhe num ícone quando o mapa
    é mexido; se estiver recolhida, o botão do ícone abre de novo."""
    if not page.evaluate(JS_ATRIBUICAO)["visivel"]:
        page.locator(".maplibregl-ctrl-attrib-button").dispatch_event("click")
        time.sleep(1.0)
    return page.evaluate(JS_ATRIBUICAO)


def aumentar_letra(page):
    """Letra da figura com LETRA_FIGURA px."""
    mudadas = page.evaluate(JS_LETRA, {"tamanho": LETRA_FIGURA})
    time.sleep(1.0)
    return mudadas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("figuras", nargs="*")
    p.add_argument("--pasta", type=Path, required=True)
    args = p.parse_intermixed_args()
    url, pedidas = args.url, args.figuras or list(FIGURAS)
    assert set(pedidas) <= set(FIGURAS), pedidas
    args.pasta.mkdir(parents=True, exist_ok=True)
    territorio = base_territorial()
    geo = dados.ler_geojson()
    with sync_playwright() as pw:
        navegador = pw.chromium.launch(channel="chrome", headless=True, args=ARGS_CHROME)
        # a densidade de pixels leva o mapa a 1600 px de largura sem mudar o enquadramento
        medida = navegador.new_page(viewport=VIEWPORT)
        medida.goto(url)
        esperar(medida)
        largura = medida.locator(".st-key-mapa").bounding_box()["width"]
        medida.close()
        print("mapa %.1f px na tela de %d; letra da figura %d px = %.1f pt a 7,8 cm"
              % (largura, VIEWPORT["width"], LETRA_FIGURA, LETRA_FIGURA * 7.8 / largura * 72 / 2.54))

        page = navegador.new_page(viewport=VIEWPORT, device_scale_factor=LARGURA_FIGURA / largura)
        page.goto(url)
        esperar(page)
        for figura in [f for f in FIGURAS if f in pedidas]:
            if figura == "risp":
                page.locator(".st-key-agregacao button", has_text="RISP").click()
                esperar(page)
            marca = mapa_parado(page)
            if figura == "risp":
                extra = "; bairros por região %s" % colorir_por_risp(page, territorio)
            else:
                principal = page.evaluate(JS_TRACOS, "choroplethmap")[0]
                extra = "; soma dos 166 = %d" % sum(principal["z"])
            destaque = esconder_destaque(page)
            # quadro e letra mudam antes dos números das AISP: o relayout redesenha o mapa e desfaz o ajuste deles
            quadro = enquadrar(page, geo)
            letras = aumentar_letra(page)
            ids = []
            if figura == "risp":
                ids, resumo = desenhar_aisp(page, territorio)
                extra += "; AISP %s" % resumo
            atribuicao = mostrar_atribuicao(page)
            creditos = atribuicao["texto"].lower()
            assert atribuicao["visivel"] and "carto" in creditos and "openstreetmap" in creditos, atribuicao
            if ids:
                na_tela = page.evaluate(JS_NUMEROS_NA_TELA, ids)
                assert len(na_tela) == 16, na_tela
                extra += "; números na tela %d" % len(na_tela)
            mapa = page.evaluate(JS_MAPA)
            assert page.evaluate(JS_MARCA_DO_MAPA) == marca, "o Streamlit refez o mapa durante os ajustes; rodar de novo"
            destino = args.pasta / FIGURAS[figura]
            page.locator(".st-key-mapa").screenshot(path=str(destino))
            print("%s: %s px%s; quadro %s; mapa %s; letra %s; destaque escondido %s; atribuição %s"
                  % (destino.name, Image.open(destino).size, extra, quadro, mapa, letras, destaque, atribuicao))
        navegador.close()


if __name__ == "__main__":
    main()
