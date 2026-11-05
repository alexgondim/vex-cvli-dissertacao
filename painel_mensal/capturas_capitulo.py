"""Imagens do painel para o capítulo, com a letra legível na largura em que entram.

Com o servidor do modo normal rodando:
python painel_mensal/capturas_capitulo.py --normal http://localhost:8765 --pasta DESTINO

O painel tem 3 abas (exploração, explicação e diagnóstico) e a seleção global na barra lateral. As figuras de
largura inteira entram no capítulo com 16 cm (textwidth): a janela do navegador tem LARGURA_JANELA px, o que deixa a
área de conteúdo com cerca de 1030 px, e a imagem recorta só essa área, com densidade 2, de modo que a letra mínima
dos gráficos, rotulos.FONTE px, fica com cerca de 6,1 pt. A barra lateral e os 2 cartões da escala própria entram
com LARGURA_ESTREITA_CM. Cada imagem sai com a menor letra medida na página, em pt, e a altura em cm na largura em
que entra, que não deve passar de ALTURA_MAXIMA_CM (com a legenda, a figura cabe numa página).

Todas com o caso inicial (Itapuã, mês-alvo de junho de 2024). Grava em DESTINO:
- aba_exploracao.png: a aba de exploração inteira, sem filtro (passa de ALTURA_MAXIMA_CM: no capítulo, só entra
  reduzida); aba_exploracao_1.png: filtros, mapa, turno e as distribuições por natureza, sexo da vítima e RISP,
  sem filtro; aba_exploracao_2.png: a série, sem filtro;
- aba_exploracao_filtrada.png: a parte 1 da aba, com o turno em madrugada e noite;
- aba_explicacao_1.png: faixa de contexto, chaves, explicação local e resumo global; aba_explicacao_2.png:
  dependência e mudança entre anos; aba_explicacao_3.png: mapa no modo contribuição (todos os atributos, por
  bairro) e série sem filtros;
- aba_diagnostico.png: a aba de diagnóstico; diagnostico_corte_fixo.png, diagnostico_janela.png,
  diagnostico_mensal.png e diagnostico_tabela.png: cada cartão dela, recortado pelo seletor dele;
- barra_lateral.png: a barra lateral, do primeiro ao último elemento, na largura dela;
- escala_propria_1.png e escala_propria_2.png: com a chave "bairro em escala própria" ligada, o cartão do resumo
  global e o da mudança entre anos, cada um sozinho;
- visao_geral_abas.png: o painel inteiro com a aba de explicação aberta e as 2 chaves desligadas, da barra lateral e
  do título até o fim dos cartões da dependência e da mudança entre anos, antes do mapa da contribuição. Depois da
  captura, uma letra em cada parte (PARTES), em círculo branco com borda preta, no canto de cima à esquerda, onde
  não cobre texto, gráfico, imagem, controle nem outra letra (o B na linha das abas, o G à esquerda do topo do
  cartão da dependência); a página não é alterada.
Com --so NOME [NOME ...], grava só esses arquivos (nome sem .png); a página que não tem arquivo pedido nem abre.
"""
import argparse
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

import demonstracao
from capturas import ARGS_CHROME, JS_ALTURA, esperar, tirar_o_mouse

LARGURA_JANELA = 1490
DENSIDADE = 2
LARGURA_CM = 16
LARGURA_ESTREITA_CM = 7.8
ALTURA_MAXIMA_CM = 21
PT_POR_CM = 72 / 2.54
FOLGA = 8
EXPLORACAO = ("aba_exploracao", "aba_exploracao_1", "aba_exploracao_2", "aba_exploracao_filtrada")
EXPLICACAO = ("aba_explicacao_1", "aba_explicacao_2", "aba_explicacao_3")
ESCALA_PROPRIA = {"escala_propria_1": ".st-key-cartao_global", "escala_propria_2": ".st-key-cartao_anos"}
VISAO_GERAL = "visao_geral_abas"
DIAGNOSTICO = {"diagnostico_corte_fixo": ".st-key-cartao_corte_fixo",
               "diagnostico_janela": ".st-key-cartao_janela", "diagnostico_mensal": ".st-key-cartao_mensal",
               "diagnostico_tabela": ".st-key-cartao_tabela"}
ARQUIVOS = {*EXPLORACAO, *EXPLICACAO, "aba_diagnostico", "barra_lateral", *ESCALA_PROPRIA, VISAO_GERAL, *DIAGNOSTICO}
# letras da visão geral: corpo de LETRA_VISAO_PT na página a 16 cm; o círculo tem 1,44 vez o corpo
LETRA_VISAO_PT = 12
RAIO_POR_CORPO = 0.72
FONTE_LETRAS = "C:/Windows/Fonts/arialbd.ttf"
PARTES = (("A", '[data-testid="stSidebar"]'), ("B", '[data-testid="stMain"] [role="tablist"]'), ("C", ".st-key-faixa"),
          ("D", ".st-key-ver_30"), ("E", ".st-key-cartao_local"), ("F", ".st-key-cartao_global"),
          ("G", ".st-key-cartao_dependencia"), ("H", ".st-key-cartao_anos"))
# o que a letra não pode cobrir: cada pedaço de texto visível, os gráficos, as imagens e os controles
JS_OBSTACULOS = """() => {
  const caixas = [];
  const vis = r => r.width > 0 && r.height > 0;
  const andar = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  while (andar.nextNode()) {
    const n = andar.currentNode; const e = n.parentElement;
    if (!n.textContent.trim() || !e || e.closest('svg')) continue;
    let op = 1; for (let a = e; a; a = a.parentElement) op *= parseFloat(getComputedStyle(a).opacity);
    if (op === 0 || getComputedStyle(e).visibility === 'hidden') continue;
    const g = document.createRange(); g.selectNodeContents(n);
    for (const r of g.getClientRects()) if (vis(r)) caixas.push([r.left, r.top, r.right, r.bottom]);
  }
  document.querySelectorAll('.js-plotly-plot, img, canvas, button, input, [role="slider"], [role="tab"], label')
    .forEach(e => { const r = e.getBoundingClientRect(); if (vis(r)) caixas.push([r.left, r.top, r.right, r.bottom]); });
  return caixas;
}"""
JS_CAIXA = """(s) => { const r = document.querySelector(s).getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]; }"""
# fim da última linha do cabeçalho ("Proposta de interface. ..."), em px de tela
JS_FIM_CABECALHO = """() => {
  const p = Array.from(document.querySelectorAll('[data-testid="stMain"] [data-testid="stMarkdownContainer"] p'))
    .find(e => e.textContent.startsWith('Proposta de interface'));
  // com os arquivos do teste não há parágrafo abaixo do título: o cabeçalho termina na última linha do título
  const alvo = p || document.querySelector('[data-testid="stMain"] h2');
  const g = document.createRange(); g.selectNodeContents(alvo); const linhas = g.getClientRects();
  return linhas[linhas.length - 1].bottom;
}"""
# menor letra de texto visível na página até a altura dada, em px de tela
JS_MENOR_LETRA_ATE = """(fim) => {
  const tamanhos = []; const dentro = r => r.width > 0 && r.height > 0 && r.bottom <= fim;
  const andar = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  while (andar.nextNode()) { const n = andar.currentNode; const e = n.parentElement;
    if (!n.textContent.trim() || !e || e.closest('svg') || !dentro(e.getBoundingClientRect())) continue;
    tamanhos.push([parseFloat(getComputedStyle(e).fontSize), n.textContent.trim().slice(0, 40)]); }
  document.querySelectorAll('svg text').forEach(e => { if (dentro(e.getBoundingClientRect()) && e.textContent.trim())
    tamanhos.push([parseFloat(getComputedStyle(e).fontSize), e.textContent.trim().slice(0, 40)]); });
  tamanhos.sort((a, b) => a[0] - b[0]);
  return tamanhos;
}"""

# caixa da barra lateral: largura inteira, altura do primeiro ao último elemento
JS_BARRA = """() => {
  const barra = document.querySelector('[data-testid="stSidebar"]');
  const blocos = Array.from(barra.querySelectorAll('[data-testid="stSidebarUserContent"] [data-testid="stElementContainer"]'))
    .map(e => e.getBoundingClientRect()).filter(b => b.height > 0);
  const r = barra.getBoundingClientRect();
  return {x0: r.left, x1: r.right, topo: Math.min(...blocos.map(b => b.top)), fim: Math.max(...blocos.map(b => b.bottom))};
}"""

# menor letra de texto visível dentro do seletor, em px de tela: texto de HTML e texto dos gráficos
JS_MENOR_LETRA = """(seletor) => {
  const raiz = document.querySelector(seletor); const tamanhos = [];
  const visivel = e => { const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
  const andar = document.createTreeWalker(raiz, NodeFilter.SHOW_TEXT);
  while (andar.nextNode()) { const n = andar.currentNode; const e = n.parentElement;
    if (!n.textContent.trim() || !e || !visivel(e) || e.closest('svg')) continue;
    tamanhos.push([parseFloat(getComputedStyle(e).fontSize), n.textContent.trim().slice(0, 40)]); }
  raiz.querySelectorAll('svg text').forEach(e => { if (visivel(e) && e.textContent.trim())
    tamanhos.push([parseFloat(getComputedStyle(e).fontSize), e.textContent.trim().slice(0, 40)]); });
  tamanhos.sort((a, b) => a[0] - b[0]);
  return tamanhos;
}"""


def ajustar_altura(page, largura=LARGURA_JANELA):
    for _ in range(3):
        altura = page.evaluate(JS_ALTURA) + 40
        if altura == page.viewport_size["height"]:
            break
        page.set_viewport_size({"width": largura, "height": altura})
        esperar(page, pausa=1.0)


def abrir(navegador, url):
    page = navegador.new_page(viewport={"width": LARGURA_JANELA, "height": 1200}, device_scale_factor=DENSIDADE)
    page.goto(url)
    esperar(page)
    ajustar_altura(page)
    return page


def abrir_aba(page, nome):
    page.get_by_role("tab", name=nome).click()
    esperar(page)
    ajustar_altura(page)


def pt_no_capitulo(px, largura_px, largura_cm=LARGURA_CM):
    return px * largura_cm / largura_px * PT_POR_CM


def pedido(so, nome):
    return so is None or nome in so


def gravar(page, clip, destino, largura_cm, seletor_letra):
    """Grava o recorte e devolve pixels, altura em cm na largura em que a figura entra e a menor letra, em pt."""
    tirar_o_mouse(page)
    esperar(page, pausa=0.5, mapa=False)
    Image.open(io.BytesIO(page.screenshot(clip=clip))).save(destino)
    tamanho = Image.open(destino).size
    altura_cm = largura_cm * tamanho[1] / tamanho[0]
    letras = page.evaluate(JS_MENOR_LETRA, seletor_letra)
    return {"arquivo": destino.name, "pixels": tamanho, "largura (cm)": largura_cm,
            "altura (cm)": round(altura_cm, 2), "cabe em %g cm" % ALTURA_MAXIMA_CM: altura_cm <= ALTURA_MAXIMA_CM,
            "menor letra (px)": letras[0][0],
            "menor letra (pt)": round(pt_no_capitulo(letras[0][0], clip["width"], largura_cm), 2),
            "textos com a menor letra": sorted({t for p, t in letras if p == letras[0][0]})[:8]}


def recortar(page, seletor, destino, largura_cm=LARGURA_CM):
    """A área do seletor, com FOLGA px de tela em volta."""
    caixa = page.locator(seletor).bounding_box()
    clip = {"x": caixa["x"] - FOLGA, "y": caixa["y"] - FOLGA, "width": caixa["width"] + 2 * FOLGA,
            "height": caixa["height"] + 2 * FOLGA}
    return gravar(page, clip, destino, largura_cm, seletor)


def recortar_barra(page, destino):
    b = page.evaluate(JS_BARRA)
    clip = {"x": b["x0"], "y": b["topo"] - FOLGA, "width": b["x1"] - b["x0"], "height": b["fim"] - b["topo"] + 2 * FOLGA}
    return gravar(page, clip, destino, LARGURA_ESTREITA_CM, '[data-testid="stSidebarUserContent"]')


def cobre(cx, cy, r, caixa):
    """O círculo encosta na caixa (x0, y0, x1, y1)?"""
    x = min(max(cx, caixa[0]), caixa[2])
    y = min(max(cy, caixa[1]), caixa[3])
    return (cx - x) ** 2 + (cy - y) ** 2 < r ** 2


def lugar_da_letra(x0, y0, r, obstaculos, largura, altura, folga=2, candidatos=None):
    """Centro do círculo junto ao canto de cima à esquerda da parte, sem encostar em nenhum obstáculo: fora do
    canto, em diagonal; à esquerda do topo; acima da borda esquerda; no próprio canto; dentro do canto. Com
    candidatos, só esses (nome, centro), e nenhum livre é erro."""
    for nome, (cx, cy) in candidatos or (("fora, em diagonal", (x0 - r - folga, y0 - r - folga)),
                                         ("à esquerda", (x0 - r - folga, y0 + r)), ("acima", (x0 + r, y0 - r - folga)),
                                         ("no canto", (x0, y0)), ("dentro", (x0 + r + folga, y0 + r + folga))):
        dentro_da_imagem = r <= cx <= largura - r and r <= cy <= altura - r
        if dentro_da_imagem and not any(cobre(cx, cy, r + folga, c) for c in obstaculos):
            return nome, cx, cy
    raise RuntimeError("sem lugar livre para a letra no canto (%.0f, %.0f)" % (x0, y0))


def visao_geral(page, destino):
    """A página como está (aba de explicação aberta, chaves desligadas), do topo até FOLGA abaixo dos cartões da
    dependência e da mudança entre anos, com a barra lateral; depois, as letras das partes."""
    tirar_o_mouse(page)
    esperar(page, pausa=0.5, mapa=False)
    largura = page.viewport_size["width"]
    cartoes = [page.locator(s).bounding_box() for s in (".st-key-cartao_dependencia", ".st-key-cartao_anos")]
    fim = max(c["y"] + c["height"] for c in cartoes) + FOLGA
    imagem = Image.open(io.BytesIO(page.screenshot(clip={"x": 0, "y": 0, "width": largura, "height": fim})))
    obstaculos = page.evaluate(JS_OBSTACULOS)
    caixas = {letra: page.evaluate(JS_CAIXA, seletor) for letra, seletor in PARTES}
    fim_cabecalho = page.evaluate(JS_FIM_CABECALHO)
    letras_painel = page.evaluate(JS_MENOR_LETRA_ATE, fim)

    px_por_pt = imagem.width / (LARGURA_CM * PT_POR_CM)
    fonte = ImageFont.truetype(FONTE_LETRAS, round(LETRA_VISAO_PT * px_por_pt))
    r = RAIO_POR_CORPO * fonte.size / DENSIDADE  # em px de tela
    desenho = ImageDraw.Draw(imagem)
    lugares = {}
    for letra, (x0, y0, _, _) in caixas.items():
        # fora do canto, o B ficava na linha do cabeçalho e o G na do subtítulo, e pareciam marcá-los: o B vai para a
        # linha das abas, no meio do espaço entre o cabeçalho e a faixa, e o G à esquerda do topo do cartão, como o E
        candidatos = None
        if letra == "B":
            candidatos = (("na linha das abas", (x0 - r - 2, (fim_cabecalho + caixas["C"][1]) / 2)),)
        elif letra == "G":
            candidatos = (("à esquerda", (x0 - r - 2, y0 + r)),)
        onde, cx, cy = lugar_da_letra(x0, y0, r, obstaculos, largura, fim, candidatos=candidatos)
        # a letra seguinte também não encosta nesta
        obstaculos.append([cx - r, cy - r, cx + r, cy + r])
        X, Y, R = cx * DENSIDADE, cy * DENSIDADE, r * DENSIDADE
        desenho.ellipse((X - R, Y - R, X + R, Y + R), fill="white", outline="black", width=max(2, round(fonte.size / 14)))
        desenho.text((X, Y), letra, font=fonte, fill="black", anchor="mm")
        lugares[letra] = {"lugar": onde, "centro (px da imagem)": [round(X), round(Y)]}
    caixa_letra = desenho.textbbox((0, 0), "H", font=fonte)
    imagem.save(destino)
    altura_cm = LARGURA_CM * imagem.height / imagem.width
    return {"arquivo": destino.name, "janela (px de tela)": list(page.viewport_size.values()), "pixels": imagem.size,
            "altura a 16 cm (cm)": round(altura_cm, 2), "cabe em %g cm" % ALTURA_MAXIMA_CM: altura_cm <= ALTURA_MAXIMA_CM,
            "largura com %g cm de altura (cm)" % ALTURA_MAXIMA_CM: round(ALTURA_MAXIMA_CM * imagem.width / imagem.height, 2),
            "menor letra do painel (px de tela)": letras_painel[0][0],
            "menor letra do painel (pt a 16 cm)": round(pt_no_capitulo(letras_painel[0][0], largura), 2),
            "letras A a H: corpo (pt a 16 cm)": round(fonte.size / px_por_pt, 2),
            "letras A a H: altura do H (pt a 16 cm)": round((caixa_letra[3] - caixa_letra[1]) / px_por_pt, 2),
            "lugares": lugares}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--normal", required=True)
    p.add_argument("--pasta", type=Path, required=True)
    p.add_argument("--so", nargs="+", metavar="NOME")
    args = p.parse_args()
    so = set(args.so) if args.so else None
    assert so is None or so <= ARQUIVOS, sorted(so - ARQUIVOS)
    args.pasta.mkdir(parents=True, exist_ok=True)
    saida = {}
    with sync_playwright() as pw:
        navegador = pw.chromium.launch(channel="chrome", headless=True, args=ARGS_CHROME)

        if any(pedido(so, n) for n in (*EXPLORACAO, "barra_lateral")):
            page = abrir(navegador, args.normal)
            if pedido(so, "aba_exploracao"):
                saida["aba_exploracao"] = recortar(page, ".st-key-aba_exploracao", args.pasta / "aba_exploracao.png")
            for i in (1, 2):
                if pedido(so, "aba_exploracao_%d" % i):
                    saida["aba_exploracao_%d" % i] = recortar(page, ".st-key-exploracao_%d" % i,
                                                              args.pasta / ("aba_exploracao_%d.png" % i))
            if pedido(so, "barra_lateral"):
                saida["barra_lateral"] = recortar_barra(page, args.pasta / "barra_lateral.png")
            if pedido(so, "aba_exploracao_filtrada"):
                for turno in ("manha", "tarde"):
                    page.locator(".st-key-turno_%s label" % turno).click()
                    esperar(page, pausa=1.0, mapa=False)
                esperar(page, pausa=0.5)
                ajustar_altura(page)
                saida["aba_exploracao_filtrada"] = recortar(page, ".st-key-exploracao_1",
                                                            args.pasta / "aba_exploracao_filtrada.png")
            page.close()

        if any(pedido(so, n) for n in (*EXPLICACAO, "aba_diagnostico", *ESCALA_PROPRIA, VISAO_GERAL, *DIAGNOSTICO)):
            page = abrir(navegador, args.normal)
            abrir_aba(page, "Explicação")
            atributo = page.locator(".st-key-atributo input").input_value()
            assert atributo == "todos os atributos", atributo
            for i, nome in enumerate(EXPLICACAO, 1):
                if pedido(so, nome):
                    saida[nome] = recortar(page, ".st-key-explicacao_%d" % i, args.pasta / (nome + ".png"))
            if pedido(so, "aba_explicacao_1"):
                # letra da cascata, que é imagem: 12 pt na figura de TAMANHO_CASCATA polegadas, mostrada na largura
                # da imagem na tela
                px = 12 * page.locator(".st-key-cascata img").bounding_box()["width"] / (demonstracao.TAMANHO_CASCATA[0] * 72)
                largura = page.locator(".st-key-explicacao_1").bounding_box()["width"] + 2 * FOLGA
                saida["aba_explicacao_1"]["cascata: menor letra (pt)"] = round(pt_no_capitulo(px, largura), 2)
            if pedido(so, VISAO_GERAL):
                saida[VISAO_GERAL] = visao_geral(page, args.pasta / (VISAO_GERAL + ".png"))
            if any(pedido(so, n) for n in ESCALA_PROPRIA):
                page.locator(".st-key-bairro_proprio label").click()
                esperar(page, pausa=1.0)
                ajustar_altura(page)
                for nome, seletor in ESCALA_PROPRIA.items():
                    if pedido(so, nome):
                        saida[nome] = recortar(page, seletor, args.pasta / (nome + ".png"), LARGURA_ESTREITA_CM)
            if any(pedido(so, n) for n in ("aba_diagnostico", *DIAGNOSTICO)):
                abrir_aba(page, "Diagnóstico")
                if pedido(so, "aba_diagnostico"):
                    saida["aba_diagnostico"] = recortar(page, ".st-key-aba_diagnostico", args.pasta / "aba_diagnostico.png")
                for nome, seletor in DIAGNOSTICO.items():
                    if pedido(so, nome):
                        saida[nome] = recortar(page, seletor, args.pasta / (nome + ".png"))
            page.close()
        navegador.close()
    print(json.dumps(saida, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
