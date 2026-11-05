"""Leitura da base de 27/09 e contagens do mock do painel mensal.

Só dado real, sem modelo. Nada aqui depende do Streamlit, para que as conferências usem as
mesmas contagens que a tela mostra.
"""
import json
import math
import unicodedata
from pathlib import Path

import pandas as pd
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

RAIZ = Path(__file__).resolve().parents[1]
PASTA_BASE = RAIZ / "analises" / "medicoes_2026-09" / "base_2018_2024"
ARQUIVOS = {
    "eventos": PASTA_BASE / "eventos.csv",
    "base_bairro_mes": PASTA_BASE / "painel_bairro_mes_features.csv",
    "dicionario": PASTA_BASE / "dicionario_painel_features.csv",
    "geojson": RAIZ / "data" / "processed" / "bairros_salvador.geojson",
}
# da base bairro-mês, só o que alguma visão exibe; a aisp é a do mapa da exploração na agregação por AISP
COLUNAS_BASE = ["bairro", "risp", "aisp", "ano_ref", "mes_ref", "ano_alvo", "mes_alvo", "n_vitimas_hist12",
                "target_cvli"]
# colunas do dicionário que ficam na base mas não entram no modelo
FORA_DO_MODELO = ("ano_ref", "mes_ref", "ano_alvo", "mes_alvo", "target_cvli")
CHAVE_POLIGONO = "NOME_BAIRR"
# propriedade dos polígonos das AISP, montados pela união dos bairros de cada uma (poligonos_aisp)
CHAVE_AISP = "aisp"
# quadro do mapa: os polígonos ocupam até 90% da altura, a 16 px da borda de cima; no mapa de 560 px sobram
# 40 px embaixo, onde o MapLibre põe a atribuição do mapa de fundo, que assim não cobre bairro
FRACAO_DA_ALTURA = 0.9
TOPO_DO_QUADRO = 16

# grafia da base -> nome do polígono no GeoJSON, nos dois bairros em que as duas diferem
EXCECAO_POLIGONO = {
    "Vila Ruy Barbosa/Jardim Cruzeiro": "Vila Ruy Barbosa\\Jardim Cruzeiro",
    "Caixa D'Água": "Caixa D´Água",
}

# valor em eventos.csv -> (rótulo na tela, faixa)
TURNOS = {
    "madrugada": ("madrugada", "00:00:00 a 05:59:59"),
    "manha": ("manhã", "06:00:00 a 11:59:59"),
    "tarde": ("tarde", "12:00:00 a 17:59:59"),
    "noite": ("noite", "18:00:00 a 23:59:59"),
}
NATUREZAS = {
    "Homicídio Doloso": "homicídio doloso",
    "Feminicídio": "feminicídio",
    "Latrocínio": "latrocínio",
    "Lesão Corporal Seguida de Morte": "lesão corporal seguida de morte",
}
# valores de sexo_vitima, na ordem da distribuição por gênero da vítima; o registro sem sexo informado fica fora dela
SEXOS = ("M", "F")
# faixas de 10 anos da distribuição por idade, da mais nova para a mais velha; a última fica aberta. Os nomes são os
# rótulos do eixo, com meia-risca
FAIXAS_IDADE = ("0–9", "10–19", "20–29", "30–39", "40–49", "50–59", "60+")
# domingo primeiro, como no esboço. Rótulos de 3 letras: com D S T Q Q S S o st.pills funde os
# rótulos repetidos e, na execução seguinte, o filtro fica com só 4 dias
DIAS = {"domingo": "dom", "segunda": "seg", "terca": "ter", "quarta": "qua", "quinta": "qui", "sexta": "sex",
        "sabado": "sáb"}
MESES = pd.period_range("2018-01", "2024-12", freq="M")
MESES_ALVO = pd.period_range("2019-01", "2024-12", freq="M")
# caso aberto ao iniciar: o item 7 pede a explicação local num mês de pico; jun/2024 tem 4 CVLI em
# Itapuã, o máximo do bairro no ano (empatado com fev/2024)
BAIRRO_INICIAL = "Itapuã"
MES_INICIAL = pd.Period("2024-06", freq="M")
_ABREV = ("jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez")


def mes_rotulo(p):
    return "%s/%d" % (_ABREV[p.month - 1], p.year)


def inteiro_br(n):
    return "{:,}".format(int(n)).replace(",", ".")


def pct_br(x, casas=1):
    return ("%.*f%%" % (casas, 100 * x)).replace(".", ",")


def rotulo_turno(turno):
    """Rótulo de eixo em 3 linhas, para caber sem inclinar: nome, início da faixa, fim da faixa."""
    nome, faixa = TURNOS[turno]
    inicio, fim = faixa.split(" a ")
    return "%s<br>%s a<br>%s" % (nome, inicio, fim)


def ordenar_bairros(nomes):
    """Ordem alfabética sem considerar acento (Águas Claras fica no começo, não depois do Z)."""
    def chave(s):
        return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return sorted(nomes, key=chave)


def ler_eventos():
    e = pd.read_csv(ARQUIVOS["eventos"])
    e["periodo"] = pd.to_datetime(e["data"]).dt.to_period("M")
    return e


def ler_base():
    return pd.read_csv(ARQUIVOS["base_bairro_mes"], usecols=COLUNAS_BASE)


def atributos_do_modelo():
    d = pd.read_csv(ARQUIVOS["dicionario"])
    nomes = [c for c in d["coluna"] if c not in FORA_DO_MODELO]
    assert len(nomes) == 30, nomes
    return nomes


def ler_geojson():
    """GeoJSON com uma só propriedade por polígono: a geometria não muda, e o que vai ao navegador encolhe."""
    geo = json.loads(ARQUIVOS["geojson"].read_text(encoding="utf-8"))
    for f in geo["features"]:
        f["properties"] = {CHAVE_POLIGONO: f["properties"][CHAVE_POLIGONO]}
    return geo


def subconjunto(geo, nomes):
    nomes = set(nomes)
    return {"type": "FeatureCollection",
            "features": [f for f in geo["features"] if f["properties"][CHAVE_POLIGONO] in nomes]}


def enquadramento(geo, nomes, altura, largura=None):
    """Centro e zoom do mapa para os polígonos de `nomes` ocuparem até FRACAO_DA_ALTURA da altura, a
    TOPO_DO_QUADRO px da borda de cima, e, com `largura`, caberem também nessa largura. O MapLibre usa
    Mercator e mede o zoom em blocos de 512 px. O app não passa a largura, porque o Streamlit estica o mapa
    até a largura do cartão."""
    nomes = set(nomes)
    lons, lats = [], []

    def percorre(c):
        if isinstance(c[0], (int, float)):
            lons.append(c[0])
            lats.append(c[1])
        else:
            for x in c:
                percorre(x)

    for f in geo["features"]:
        if f["properties"][CHAVE_POLIGONO] in nomes:
            percorre(f["geometry"]["coordinates"])

    def mercator(lat):
        return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

    px_por_radiano = FRACAO_DA_ALTURA * altura / (mercator(max(lats)) - mercator(min(lats)))
    if largura:
        px_por_radiano = min(px_por_radiano, largura / math.radians(max(lons) - min(lons)))
    y_centro = mercator(max(lats)) - (altura / 2 - TOPO_DO_QUADRO) / px_por_radiano
    centro = {"lat": math.degrees(2 * math.atan(math.exp(y_centro)) - math.pi / 2), "lon": (min(lons) + max(lons)) / 2}
    return centro, math.log2(px_por_radiano * 2 * math.pi / 512)


def poligono(bairro):
    return EXCECAO_POLIGONO.get(bairro, bairro)


def poligonos_sem_bairro(geo, bairros):
    usados = {poligono(b) for b in bairros}
    return sorted(f["properties"][CHAVE_POLIGONO] for f in geo["features"]
                  if f["properties"][CHAVE_POLIGONO] not in usados)


def bairros_sem_poligono(geo, bairros):
    nomes = {f["properties"][CHAVE_POLIGONO] for f in geo["features"]}
    return sorted(b for b in bairros if poligono(b) not in nomes)


def risp_por_bairro(base):
    assert (base.groupby("bairro")["risp"].nunique() == 1).all()
    return base.groupby("bairro")["risp"].first()


def aisp_por_bairro(base):
    """AISP de cada bairro, a da base de 27/09 (Portaria SSP 435/2020, Anexo I), uma por bairro."""
    assert (base.groupby("bairro")["aisp"].nunique() == 1).all()
    return base.groupby("bairro")["aisp"].first().astype(int)


def poligonos_aisp(geo, aisp_de):
    """Um polígono por AISP, a união dos polígonos dos bairros dela. Geometria inválida, de bairro ou da união, é
    corrigida só para o desenho, com buffer(0); devolve também a lista do que precisou."""
    formas = {f["properties"][CHAVE_POLIGONO]: shape(f["geometry"]) for f in geo["features"]}
    corrigidos, features = [], []
    for aisp in sorted(aisp_de.unique()):
        partes = []
        for b in aisp_de.index[aisp_de == aisp]:
            g = formas[poligono(b)]
            if not g.is_valid:
                g = g.buffer(0)
                corrigidos.append(b)
            partes.append(g)
        uniao = unary_union(partes)
        if not uniao.is_valid:
            uniao = uniao.buffer(0)
            corrigidos.append("AISP %d" % aisp)
        features.append({"type": "Feature", "properties": {CHAVE_AISP: int(aisp)}, "geometry": mapping(uniao)})
    return {"type": "FeatureCollection", "features": features}, corrigidos


def selecionados(selecao):
    """Seleção de bairros como lista: aceita um nome só (as conferências e figuras antigas passam o bairro)."""
    if selecao is None:
        return []
    return [selecao] if isinstance(selecao, str) else list(selecao)


def filtrar(e, periodo, naturezas, turnos, dias):
    """Filtros exploratórios. `turnos=None` deixa todos os turnos (usado na distribuição por turno), e
    `naturezas=None`, todas as naturezas (usado na distribuição por natureza)."""
    m = (e["periodo"] >= periodo[0]) & (e["periodo"] <= periodo[1])
    m &= e["dia_semana"].isin(dias)
    if naturezas is not None:
        m &= e["tipo_crime"].isin(naturezas)
    if turnos is not None:
        m &= e["turno"].isin(turnos)
    return e[m]


def contagem_por_bairro(e_filtrado, bairros):
    return e_filtrado.groupby("bairro").size().reindex(bairros, fill_value=0)


def tabela_mapa(contagem, risp_de, agregacao):
    t = pd.DataFrame({"bairro": contagem.index, "registros": contagem.to_numpy()})
    t["poligono"] = t["bairro"].map(poligono)
    t["risp"] = t["bairro"].map(risp_de)
    t["valor"] = t.groupby("risp")["registros"].transform("sum") if agregacao == "risp" else t["registros"]
    return t


def tabela_aisp(contagem, aisp_de):
    """Contagem por AISP: a soma dos bairros de cada uma, nas 16."""
    t = pd.DataFrame({"aisp": contagem.index.map(aisp_de), "registros": contagem.to_numpy()})
    t = t.groupby("aisp", as_index=False)["registros"].sum()
    return t.assign(valor=t["registros"])


def escopos(recorte, selecao):
    """Os recortes das distribuições: a seleção de bairros (escopo "bairro"), se houver, e Salvador inteira."""
    sel = selecionados(selecao)
    return ([("bairro", recorte[recorte["bairro"].isin(sel)])] if sel else []) + [("cidade", recorte)]


def serie_mensal(e_filtrado, selecao, periodo):
    """Registros por mês da soma dos bairros selecionados; sem seleção, de Salvador inteira."""
    meses = pd.period_range(periodo[0], periodo[1], freq="M")
    sel = selecionados(selecao)
    recorte = e_filtrado[e_filtrado["bairro"].isin(sel)] if sel else e_filtrado
    return recorte.groupby("periodo").size().reindex(meses, fill_value=0)


def distribuicao_turno(e_sem_filtro_de_turno, selecao):
    """Contagem e fração dos 4 turnos no período filtrado, na seleção de bairros (se houver) e na cidade.

    O denominador é o total dos 4 turnos com período, natureza e dia da semana aplicados; o filtro
    de turno não entra aqui (na tela ele só esmaece as barras desmarcadas).
    """
    linhas, totais = [], {}
    for escopo, sub in escopos(e_sem_filtro_de_turno, selecao):
        cont = sub.groupby("turno").size().reindex(list(TURNOS), fill_value=0)
        total = int(cont.sum())
        totais[escopo] = total
        for t in TURNOS:
            linhas.append({"escopo": escopo, "turno": t, "registros": int(cont[t]),
                           "fracao": cont[t] / total if total else 0.0})
    return pd.DataFrame(linhas), totais


def _distribuicao(recorte, selecao, coluna, categorias):
    """Contagem e fração de cada categoria de `coluna` no recorte, na seleção de bairros (se houver) e na cidade. Só
    as categorias dadas entram no denominador; com o recorte vazio, as frações ficam em 0."""
    linhas, totais = [], {}
    for escopo, sub in escopos(recorte, selecao):
        cont = sub[coluna].value_counts().reindex(list(categorias), fill_value=0)
        total = int(cont.sum())
        totais[escopo] = total
        for c in categorias:
            linhas.append({"escopo": escopo, "categoria": c, "registros": int(cont[c]),
                           "fracao": cont[c] / total if total else 0.0})
    return pd.DataFrame(linhas), totais


def distribuicao_natureza(e_sem_filtro_de_natureza, selecao):
    """Como a do turno: o denominador é o total das 4 naturezas com período, turno e dia da semana aplicados; o
    filtro de natureza não entra (na tela ele só esmaece as barras desmarcadas)."""
    return _distribuicao(e_sem_filtro_de_natureza, selecao, "tipo_crime", NATUREZAS)


def distribuicao_sexo(e_filtrado, selecao):
    """Com todos os filtros. O registro sem sexo informado fica fora das barras e do denominador."""
    return _distribuicao(e_filtrado, selecao, "sexo_vitima", SEXOS)


def distribuicao_idade(e_filtrado, selecao):
    """Com todos os filtros, em faixas de 10 anos (FAIXAS_IDADE). O registro sem idade informada fica fora das barras
    e do denominador: na base, a idade ausente já vem vazia, inclusive nos 51 registros de 2018 que a fonte gravava
    com idade 0; os zeros dos outros anos são idade informada e entram na primeira faixa."""
    com_idade = e_filtrado[e_filtrado["idade_vitima"].notna()]
    ordem = (com_idade["idade_vitima"] // 10).clip(upper=len(FAIXAS_IDADE) - 1).astype(int)
    return _distribuicao(com_idade.assign(faixa_idade=ordem.map(dict(enumerate(FAIXAS_IDADE)))), selecao,
                         "faixa_idade", FAIXAS_IDADE)


def distribuicao_risp(e_filtrado, risp_de, ordem):
    """Contagem e fração de cada RISP em Salvador, com todos os filtros. A RISP de cada registro é a do bairro
    dele (risp_de), a mesma do mapa na agregação por RISP."""
    cont = e_filtrado["bairro"].map(risp_de).value_counts().reindex(list(ordem), fill_value=0)
    total = int(cont.sum())
    return pd.DataFrame({"risp": list(ordem), "registros": [int(n) for n in cont],
                         "fracao": [n / total if total else 0.0 for n in cont]}), total


def linha_da_base(base, bairro, periodo_alvo):
    sel = base[(base["bairro"] == bairro) & (base["ano_alvo"] == periodo_alvo.year)
               & (base["mes_alvo"] == periodo_alvo.month)]
    assert len(sel) == 1, (bairro, periodo_alvo, len(sel))
    return sel.iloc[0]


def mes_de_referencia(linha):
    return pd.Period(year=int(linha["ano_ref"]), month=int(linha["mes_ref"]), freq="M")
