"""Capturas do mock e leitura do que a tela mostra, para as conferências.

Com o app rodando (./venv/Scripts/python.exe -m streamlit run painel_mensal/app.py):
python painel_mensal/capturas.py http://localhost:8501 --pasta DESTINO [--so 23] [textos_da_tela.json]
As 4 capturas têm largura 1600: as 3 primeiras são recortes da página, e a 4ª é só o cartão do mapa,
ampliado pela densidade de pixels. O destino não tem padrão, para uma execução nova não sobrescrever
capturas já entregues. Com --so, grava só as capturas pedidas (23: a 2 e a 3); as conferências na
tela rodam sempre. O arquivo de textos, opcional, grava o texto da tela em cada estado visitado,
para a busca de termos.
"""
import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
from PIL import Image
from playwright.sync_api import sync_playwright

import dados

LARGURA = 1600
# o mapa usa WebGL; sem GPU, o Chrome sem janela precisa do renderizador por software
ARGS_CHROME = ["--enable-unsafe-swiftshader", "--use-angle=swiftshader", "--ignore-gpu-blocklist"]
# as 2 linhas abaixo da explicação local são os únicos lugares da tela com o n_vitimas_hist12 e com
# o target_cvli do mês-alvo
RE_N = r"calculado sobre (\d+) vítima"
RE_OBSERVADO = r"Observado em (\S+): (\d+ registros?)\."

JS_TEXTO = """() => document.body.innerText + '\\n' +
  Array.from(document.querySelectorAll('svg text')).map(t => t.textContent).join('\\n')"""
JS_ALTURA = """() => {
  const partes = ['[data-testid="stMain"]', '[data-testid="stSidebarContent"]']
    .map(s => document.querySelector(s)).filter(Boolean).map(e => e.scrollHeight);
  return Math.max(document.documentElement.scrollHeight, ...partes);
}"""
JS_TRACOS = """(tipo) => {
  const g = Array.from(document.querySelectorAll('.js-plotly-plot'))
    .find(d => d._fullData && d._fullData.length && d._fullData[0].type === tipo);
  if (!g) return null;
  const lista = v => (v === undefined || v === null) ? null : (typeof v !== 'object') ? v
    : Array.from(v, x => (typeof x === 'object' && x !== null) ? Array.from(x) : x);
  return g._fullData.map(t => ({name: t.name, locations: lista(t.locations), z: lista(t.z), x: lista(t.x),
                                y: lista(t.y), text: lista(t.text), customdata: lista(t.customdata),
                                opacidade: lista(t.marker ? t.marker.opacity : null)}));
}"""
JS_HOVER = "() => Array.from(document.querySelectorAll('.hoverlayer .hovertext')).map(h => h.textContent)"
# rótulos do mapa de fundo, que são do MapLibre e ficam fora do alcance da figura: nas imagens do
# capítulo os nomes de cidade pequena ficariam abaixo de 6 pt. Na página da captura, a letra deles cresce
# FATOR_ROTULOS_DO_FUNDO vezes, e os de rio, lago e vila, que nem assim chegariam a 6 pt, saem
FATOR_ROTULOS_DO_FUNDO = 1.25
ROTULOS_DO_FUNDO_QUE_SAEM = ["waterway_label", "watername_lake", "watername_lake_line", "place_villages"]
JS_ROTULOS_DO_FUNDO = """(a) => {
  const gd = Array.from(document.querySelectorAll('.js-plotly-plot'))
    .find(d => d._fullData && d._fullData.length && d._fullData[0].type === 'choroplethmap');
  const map = gd._fullLayout.map._subplot.map;
  const mudadas = [], escondidas = [];
  map.getStyle().layers.forEach(l => {
    if (l.type !== 'symbol' || l.id.startsWith('plotly') || !l.layout || l.layout['text-size'] === undefined) return;
    if (a.saem.includes(l.id)) { map.setLayoutProperty(l.id, 'visibility', 'none'); escondidas.push(l.id); return; }
    let ts = map.getLayoutProperty(l.id, 'text-size');
    if (typeof ts === 'number') ts = ts * a.fator;
    else if (ts && ts.stops) ts = Object.assign({}, ts, {stops: ts.stops.map(([z, v]) => [z, v * a.fator])});
    else if (Array.isArray(ts) && ts[0] === 'interpolate') ts = ts.map((x, i) => (i > 3 && i % 2 === 0) ? x * a.fator : x);
    else return;
    map.setLayoutProperty(l.id, 'text-size', ts);
    mudadas.push(l.id);
  });
  return {mudadas: mudadas, escondidas: escondidas};
}"""
# tira nomes do mapa de fundo por filtro nas camadas de nome de lugar, e lista os nomes que ficam na tela
JS_SEM_NOMES_DO_FUNDO = """(nomes) => {
  const gd = Array.from(document.querySelectorAll('.js-plotly-plot'))
    .find(d => d._fullData && d._fullData.length && d._fullData[0].type === 'choroplethmap');
  const map = gd._fullLayout.map._subplot.map;
  const fora = nomes.flatMap(n => [['!=', 'name_en', n], ['!=', 'name', n]]);
  map.getStyle().layers.filter(l => l.type === 'symbol' && l.id.startsWith('place_'))
    .forEach(l => map.setFilter(l.id, ['all', map.getFilter(l.id) || ['has', 'name'], ...fora]));
}"""
JS_NOMES_DO_FUNDO = """() => {
  const gd = Array.from(document.querySelectorAll('.js-plotly-plot'))
    .find(d => d._fullData && d._fullData.length && d._fullData[0].type === 'choroplethmap');
  const map = gd._fullLayout.map._subplot.map;
  const camadas = map.getStyle().layers.filter(l => l.type === 'symbol' && l.id.startsWith('place_')).map(l => l.id);
  return [...new Set(map.queryRenderedFeatures({layers: camadas}).map(f => f.properties.name_en || f.properties.name))];
}"""


def esperar(page, pausa=1.5, mapa=True):
    time.sleep(pausa)
    page.wait_for_selector('[data-testid="stApp"][data-test-script-state="notRunning"]', timeout=180000)
    page.wait_for_function("() => document.querySelectorAll('.js-plotly-plot').length >= 5", timeout=180000)
    erros = page.locator('[data-testid="stException"]')
    if erros.count():
        raise RuntimeError("exceção na tela: " + erros.first.inner_text())
    if mapa:
        time.sleep(6)  # tiles do mapa de fundo


def recorte(page, seletor_topo, seletor_fim, do_topo_da_pagina=False):
    topo = 0 if do_topo_da_pagina else page.locator(seletor_topo).bounding_box()["y"] - 16
    caixa = page.locator(seletor_fim).bounding_box()
    return {"x": 0, "y": topo, "width": LARGURA, "height": caixa["y"] + caixa["height"] + 16 - topo}


def bairro_na_tela(page):
    """Bairro selecionado, lido do título da série mensal."""
    titulo = page.locator("h5", has_text="Série mensal de").first.inner_text()
    return titulo.replace("Série mensal de", "").strip()


def n_na_tela(page):
    return [int(v) for v in re.findall(RE_N, page.locator(".st-key-linhas_bc").inner_text())]


def observado_na_tela(page):
    return re.findall(RE_OBSERVADO, page.locator(".st-key-linhas_bc").inner_text())


def resumo_barras(tracos):
    """Legenda da distribuição por turno e, por turno, o rótulo, o texto do hover e a opacidade de cada barra."""
    return [{"legenda": t["name"],
             "barras": {x.split("<br>")[0]: {"rótulo": texto, "hover": registros, "opacidade": op}
                        for x, texto, registros, op in zip(t["x"], t["text"], t["customdata"], t["opacidade"])}}
            for t in tracos]


def passar_o_mouse(page, fx, fy):
    """Passa o mouse sobre um ponto do mapa (frações da largura e da altura) e devolve o ponto."""
    caixa = page.locator(".st-key-mapa").bounding_box()
    x, y = caixa["x"] + fx * caixa["width"], caixa["y"] + fy * caixa["height"]
    page.mouse.move(x - 5, y - 5)
    time.sleep(0.3)
    page.mouse.move(x, y)
    time.sleep(1.0)
    return x, y


def escolher(page, chave, texto):
    """Escolhe uma opção num selectbox da barra lateral digitando o texto dela."""
    page.locator(".st-key-%s" % chave).click()
    page.keyboard.type(texto)
    time.sleep(1.0)
    page.keyboard.press("Enter")
    esperar(page)


def aumentar_rotulos_do_fundo(page, fator=FATOR_ROTULOS_DO_FUNDO, saem=ROTULOS_DO_FUNDO_QUE_SAEM):
    resultado = page.evaluate(JS_ROTULOS_DO_FUNDO, {"fator": fator, "saem": list(saem)})
    time.sleep(2.0)  # o mapa refaz os rótulos
    return resultado


def tirar_nomes_do_fundo(page, nomes):
    """Tira os nomes do mapa de fundo e devolve os que ficam na tela."""
    page.evaluate(JS_SEM_NOMES_DO_FUNDO, list(nomes))
    time.sleep(2.0)
    ficam = page.evaluate(JS_NOMES_DO_FUNDO)
    assert not set(nomes) & set(ficam), ficam
    return ficam


def tirar_o_mouse(page):
    """Leva o mouse ao canto de baixo da página, longe de qualquer controle: com ele sobre a barra
    lateral, por exemplo, aparece o botão de recolhê-la."""
    page.mouse.move(page.viewport_size["width"] - 20, page.viewport_size["height"] - 20)


def ajustar_altura(page):
    for _ in range(3):
        altura = page.evaluate(JS_ALTURA) + 40
        if altura == page.viewport_size["height"]:
            break
        page.set_viewport_size({"width": LARGURA, "height": altura})
        esperar(page, pausa=1.0)


def cartao_ampliado(navegador, url, page, destino):
    """Cartão do mapa no modo contribuição com LARGURA px de largura.

    Numa página nova com a mesma janela de `page`, a densidade de pixels amplia o cartão sem mudar
    o enquadramento do mapa, como em figuras_mapa.py.
    """
    largura = page.locator(".st-key-cartao_mapa").bounding_box()["width"]
    ampliada = navegador.new_page(viewport=page.viewport_size, device_scale_factor=LARGURA / largura)
    ampliada.goto(url)
    esperar(ampliada)
    ampliada.locator(".st-key-modo button", has_text="contribuição").click()
    tirar_o_mouse(ampliada)
    esperar(ampliada)
    ampliada.locator(".st-key-cartao_mapa").screenshot(path=str(destino))
    ampliada.close()
    return {"cartão na tela (px)": largura, "arquivo (px)": Image.open(destino).size}


def argumentos():
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("textos", nargs="?", type=Path, help="grava o texto da tela em cada estado visitado")
    p.add_argument("--pasta", type=Path, required=True, help="destino das capturas")
    p.add_argument("--so", default="1234", help="capturas a gravar, por exemplo 23 para só a 2 e a 3")
    # intermixed: o arquivo de textos pode vir depois de --pasta, na ordem da docstring
    return p.parse_intermixed_args()


def main():
    args = argumentos()
    url, pasta, pedidas = args.url, args.pasta, set(args.so)
    assert pedidas <= set("1234"), args.so
    pasta.mkdir(parents=True, exist_ok=True)
    geo = dados.ler_geojson()
    base = dados.ler_base()

    def da_base(periodo_alvo, coluna):
        return int(dados.linha_da_base(base, dados.BAIRRO_INICIAL, periodo_alvo)[coluna])

    def conferencia_2(page, periodo_alvo):
        return {"mês": dados.mes_rotulo(periodo_alvo),
                "n_vitimas_hist12 na base": da_base(periodo_alvo, "n_vitimas_hist12"),
                "n abaixo da explicação local": n_na_tela(page),
                "target_cvli na base": da_base(periodo_alvo, "target_cvli"),
                "observado na tela": observado_na_tela(page)}

    textos, saida = {}, {}

    with sync_playwright() as p:
        navegador = p.chromium.launch(channel="chrome", headless=True, args=ARGS_CHROME)
        page = navegador.new_page(viewport={"width": LARGURA, "height": 1200}, device_scale_factor=1)
        page.goto(url)
        esperar(page)
        saida["webgl"] = page.evaluate("() => !!document.createElement('canvas').getContext('webgl')")
        ajustar_altura(page)

        # estado inicial: dados.BAIRRO_INICIAL em dados.MES_INICIAL, modo observado, agregação por bairro, sem filtro
        textos["inicial"] = page.evaluate(JS_TEXTO)
        principal, sem_registro, contorno = page.evaluate(JS_TRACOS, "choroplethmap")
        nomes_geo = {f["properties"][dados.CHAVE_POLIGONO] for f in geo["features"]}
        saida["conf1_soma_bairros"] = sum(principal["z"])
        saida["conf3"] = {
            "coloridos": len(principal["locations"]),
            "sem_cor": sem_registro["locations"],
            "legenda na tela": "□ sem registro na base" in textos["inicial"],
            "polígonos cobertos": len(set(principal["locations"]) | set(sem_registro["locations"])),
            "polígonos no geojson": len(nomes_geo),
            "sem correspondência": sorted(nomes_geo - set(principal["locations"]) - set(sem_registro["locations"])),
            "contorno": contorno["locations"],
        }
        saida["conf2"] = conferencia_2(page, dados.MES_INICIAL)
        observado = page.evaluate(JS_TRACOS, "scatter")[0]
        saida["serie_2024_na_tela"] = {m: y for m, y in zip(observado["customdata"], observado["y"])
                                       if m.endswith("/2024")}
        saida["legendas_da_serie"] = [t["name"] for t in page.evaluate(JS_TRACOS, "scatter")]
        saida["abas"] = page.get_by_role("tab").all_inner_texts()
        if "1" in pedidas:
            page.screenshot(path=str(pasta / "1_linha_a_observado_sem_filtro.png"),
                            clip=recorte(page, None, ".st-key-linha_a", do_topo_da_pagina=True))

        page.locator(".st-key-agregacao button", has_text="RISP").click()
        esperar(page)
        textos["agregacao_risp"] = page.evaluate(JS_TEXTO)
        principal = page.evaluate(JS_TRACOS, "choroplethmap")[0]
        por_risp = {}
        for _, texto in principal["customdata"]:
            risp, n = re.search(r"RISP (.+?): ([\d.]+) registro", texto).groups()
            por_risp[risp] = int(n.replace(".", ""))
        saida["conf1_por_risp"] = por_risp
        saida["conf1_soma_risp"] = sum(por_risp.values())
        passar_o_mouse(page, 0.53, 0.55)
        saida["hover_risp"] = page.evaluate(JS_HOVER)

        page.reload()
        esperar(page)
        page.locator(".st-key-modo button", has_text="contribuição").click()
        esperar(page)
        textos["modo_contribuicao"] = page.evaluate(JS_TEXTO)
        saida["contribuicao"] = {
            "seletor de atributo desabilitado": page.locator(".st-key-atributo input").is_disabled(),
            "valor do seletor": page.locator(".st-key-atributo input").input_value(),
            "opção por bloco na tela": "por bloco" in textos["modo_contribuicao"],
            "polígonos em cinza": len(page.evaluate(JS_TRACOS, "choroplethmap")[0]["locations"]),
        }
        if "4" in pedidas:
            saida["contribuicao"]["captura"] = cartao_ampliado(navegador, url, page,
                                                               pasta / "4_cartao_mapa_contribuicao.png")

        page.reload()
        esperar(page)
        for turno in ("manha", "tarde"):
            page.locator(".st-key-turno_%s label" % turno).click()
            esperar(page, pausa=1.0, mapa=False)
        tirar_o_mouse(page)
        esperar(page, pausa=0.5)
        textos["turno_madrugada_noite"] = page.evaluate(JS_TEXTO)
        saida["captura2_turno"] = resumo_barras(page.evaluate(JS_TRACOS, "bar"))
        if "2" in pedidas:
            page.screenshot(path=str(pasta / "2_linha_a_turno_madrugada_noite.png"),
                            clip=recorte(page, None, ".st-key-linha_a", do_topo_da_pagina=True))

        page.locator(".st-key-turno_noite label").click()
        esperar(page)
        textos["turno_so_madrugada"] = page.evaluate(JS_TEXTO)
        saida["conf4_turno"] = resumo_barras(page.evaluate(JS_TRACOS, "bar"))
        saida["conf4_soma_do_mapa"] = sum(page.evaluate(JS_TRACOS, "choroplethmap")[0]["z"])

        page.reload()
        esperar(page)
        saida["captura3"] = conferencia_2(page, dados.MES_INICIAL)
        if "3" in pedidas:
            page.screenshot(path=str(pasta / "3_linhas_b_c.png"),
                            clip=recorte(page, ".st-key-linhas_bc", ".st-key-linhas_bc"))

        page.get_by_role("tab", name="Diagnóstico").click()
        esperar(page, pausa=1.0, mapa=False)
        textos["aba_diagnostico"] = page.evaluate(JS_TEXTO)

        # clique no mapa: o plotly só reconhece o polígono depois de o mouse passar sobre ele
        page.reload()
        esperar(page)
        x, y = passar_o_mouse(page, 0.53, 0.55)
        page.mouse.click(x, y)
        esperar(page)
        saida["clique"] = {"depois do clique no mapa": bairro_na_tela(page),
                           "contorno": page.evaluate(JS_TRACOS, "choroplethmap")[2]["locations"]}
        escolher(page, "sel_bairro", dados.BAIRRO_INICIAL)
        saida["clique"]["depois de escolher %s na lista" % dados.BAIRRO_INICIAL] = bairro_na_tela(page)

        # conferência 2 também no mês do despacho de 28/09 (mar/2024, sem registro no mês) e num mês
        # com 1 registro (jan/2024), para o singular, escolhidos na lista
        for periodo in (pd.Period("2024-03", freq="M"), pd.Period("2024-01", freq="M")):
            escolher(page, "sel_mes", dados.mes_rotulo(periodo))
            saida["conf2_%s" % dados.mes_rotulo(periodo)] = conferencia_2(page, periodo)
        navegador.close()

    print(json.dumps(saida, ensure_ascii=False, indent=1))
    if args.textos:
        args.textos.write_text(json.dumps(textos, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
