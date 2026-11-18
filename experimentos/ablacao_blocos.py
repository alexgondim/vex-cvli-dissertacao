"""Ablação por blocos de atributos: M1 criminal, M2 com o temporal, M3 com o espacial e M4 com o perfil (os 30
atributos), todos com a configuração de `resultados/escolha.json` e a semente 42, sem busca.

A composição dos blocos é a deste programa (criminal 16, temporal 8, espacial 3, perfil 3): as 4 proporções de turno
e de fim de semana ficam no temporal. Lê a base por bairro e mês, pelo `experimentos/validacao_comum.py`.

Dois modos, nesta ordem:
- `validacao`: as 12 dobras de 2023 (treino com todo alvo anterior ao mês, desde jan/2019; previsão das 166 linhas do
  mês), sem as linhas de alvo em 2024. Grava `ablacao_blocos_2023.csv` (métricas por dobra, média das 12 dobras e as
  1.992 linhas juntas) e `previsoes_ablacao_blocos2_2023.csv`;
- `teste`: corte fixo, com treino nas 7.968 linhas de alvo de 2019 a 2022 e avaliação nas 1.992 de 2024. Grava
  `ablacao_blocos_2024.csv`, `previsoes_ablacao_blocos2_2024.csv` e `ablacao_blocos_resumo.json` (os 4 modelos lado a
  lado e o ganho de cada bloco), que lê também o `ablacao_blocos_2023.csv`.

Grava em `resultados/`. Da raiz do repositório:
python experimentos/ablacao_blocos.py validacao
python experimentos/ablacao_blocos.py teste
"""
import sys

sys.dont_write_bytecode = True  # o import dos módulos comuns não deixa bytecode

import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RODADA = Path(__file__).resolve().parent
sys.path.insert(0, str(RODADA))
import validacao_comum as c  # noqa: E402

comum = c.comum
ALVO = comum.ALVO
RESULTADOS = c.RESULTADOS
SAIDA = {2023: RESULTADOS / "ablacao_blocos_2023.csv", 2024: RESULTADOS / "ablacao_blocos_2024.csv"}
PREVISOES = {2023: RESULTADOS / "previsoes_ablacao_blocos2_2023.csv", 2024: RESULTADOS / "previsoes_ablacao_blocos2_2024.csv"}
RESUMO = RESULTADOS / "ablacao_blocos_resumo.json"

# Composição dos blocos da ablação. Difere da de comum.BLOCOS: as 4 proporções de turno e de fim de semana saem do
# perfil e entram no temporal; criminal e espacial não mudam.
BLOCOS = {
    "criminal": ["cvli_lag%d" % k for k in range(1, 13)] + ["cvli_ma3", "cvli_ma6", "cvli_ma12", "cvli_trend"],
    "temporal": ["month_sin_alvo", "month_cos_alvo", "year_norm_alvo", "time_index",
                 "prop_madrugada_hist12", "prop_tarde_hist12", "prop_noite_hist12", "prop_fds_hist12"],
    "espacial": ["bairro", "aisp", "risp"],
    "perfil": ["prop_masc_hist12", "idade_media_hist12", "n_vitimas_hist12"],
}
TAMANHO = {"criminal": 16, "temporal": 8, "espacial": 3, "perfil": 3}
MODELOS = {"M1": ["criminal"], "M2": ["criminal", "temporal"], "M3": ["criminal", "temporal", "espacial"],
           "M4": ["criminal", "temporal", "espacial", "perfil"]}
BLOCO_ACRESCENTADO = {"M1": "criminal", "M2": "temporal", "M3": "espacial", "M4": "perfil"}
TAMANHO_DO_MODELO = {"M1": 16, "M2": 24, "M3": 27, "M4": 30}
MOVIDAS = ["prop_madrugada_hist12", "prop_tarde_hist12", "prop_noite_hist12", "prop_fds_hist12"]

METRICAS = ("mae", "rmse", "deviance")
PASSOS = ("validacao", "teste")


def agora():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def registrar(texto):
    linha = "%s  %s" % (agora(), texto)
    print(linha, flush=True)


def seis(v):
    return "%.6f" % v


def conferir_blocos():
    """Antes de treinar: blocos sem atributo repetido, com 16, 8, 3 e 3 atributos, que juntos são os 30 do
    dicionário; criminal e espacial iguais aos de comum.BLOCOS; temporal = o de comum.BLOCOS mais as 4 proporções, e
    perfil = o de comum.BLOCOS menos elas. Devolve os atributos de cada modelo, na ordem do dicionário."""
    todos = sum(BLOCOS.values(), [])
    assert len(todos) == len(set(todos)) == 30, todos
    assert sorted(todos) == sorted(comum.atributos())
    assert {b: len(a) for b, a in BLOCOS.items()} == TAMANHO
    antigo = comum.BLOCOS
    assert BLOCOS["criminal"] == antigo["criminal"] and BLOCOS["espacial"] == antigo["espacial"]
    assert sorted(BLOCOS["temporal"]) == sorted(antigo["temporal"] + MOVIDAS)
    assert sorted(BLOCOS["perfil"]) == sorted(set(antigo["perfil"]) - set(MOVIDAS))
    nomes = {m: comum.em_ordem(sum((BLOCOS[b] for b in blocos), [])) for m, blocos in MODELOS.items()}
    assert {m: len(a) for m, a in nomes.items()} == TAMANHO_DO_MODELO
    assert nomes["M1"] == comum.em_ordem(antigo["criminal"]) and nomes["M4"] == comum.atributos()
    return nomes


def corte_fixo(base, ano, modelos, nomes):
    """Como no experimentos/sensibilidade_corte_fixo.py: cada modelo treinado nas 7.968 linhas de alvo de 2019 a 2022
    e avaliado nas 1.992 linhas do ano."""
    treino = base[base["ano_alvo"].between(*c.ANOS_DE_TREINO)]
    avaliadas = base[base["ano_alvo"] == ano]
    assert len(treino) == 7968 and len(avaliadas) == 1992
    y = avaliadas[ALVO].to_numpy(float)
    chave = avaliadas[["bairro", "ano_alvo", "mes_alvo"]].assign(bairro=avaliadas["bairro"].astype(str))
    metricas, partes = {}, []
    for m in modelos:
        modelo = comum.estimador(c.CONFIG).fit(treino[nomes[m]], treino[ALVO].to_numpy(float))
        q = modelo.predict(avaliadas[nomes[m]]).astype(float)
        metricas[m] = comum.metricas(y, q)
        partes.append(chave.assign(modelo=m, alvo=y, previsao=q))
    return metricas, pd.concat(partes, ignore_index=True)


def tabela_do_teste(metricas, nomes):
    return pd.DataFrame([{"modelo": m, "atributos": len(nomes[m]), "linhas": 1992, **metricas[m]} for m in MODELOS])


def resumo_de(v2023, t2024, nomes):
    """Os 4 modelos lado a lado em 2023 (média das 12 dobras e 1.992 linhas) e em 2024, e o ganho de cada
    bloco, que é a métrica do modelo com o bloco menos a do modelo anterior."""
    anos = {"2023_media_das_12_dobras": v2023[v2023["dobra"] == "media_das_12_dobras"].set_index("modelo"),
            "2023_todas_as_1992_linhas": v2023[v2023["dobra"] == "todas_as_1992_linhas"].set_index("modelo"),
            "2024_corte_fixo": t2024.set_index("modelo")}
    for t in anos.values():
        assert sorted(t.index) == sorted(MODELOS)
    pares = list(zip(list(MODELOS)[:-1], list(MODELOS)[1:]))
    return {"composicao": BLOCOS,
            "modelos": {m: {"blocos": MODELOS[m], "atributos": len(nomes[m])} for m in MODELOS},
            "configuracao": c.CONFIG, "semente": comum.SEMENTE,
            "metricas": {a: {m: {k: float(t.loc[m, k]) for k in METRICAS} for m in MODELOS} for a, t in anos.items()},
            "ganho_do_bloco": {a: {BLOCO_ACRESCENTADO[m]: {k: float(t.loc[m, k] - t.loc[ant, k]) for k in METRICAS}
                                   for ant, m in pares} for a, t in anos.items()},
            "leitura_do_ganho": "métrica do modelo com o bloco menos a do modelo anterior; negativo é melhora"}


def validacao():
    """As 12 dobras de 2023 para os 4 modelos."""
    registrar("validação: início")
    base = c.ler_base("ensaio")
    assert base["ano_alvo"].max() == 2023
    registrar("validação: as linhas de alvo em 2024 saíram logo depois da leitura; ficam %d linhas, com alvo de %d "
              "a %d" % (len(base), base["ano_alvo"].min(), base["ano_alvo"].max()))
    nomes = conferir_blocos()
    registrar("blocos conferidos: criminal 16, temporal 8, espacial 3, perfil 3, sem repetição, somando os 30 do "
              "dicionário; M1 %d, M2 %d, M3 %d, M4 %d atributos"
              % tuple(len(nomes[m]) for m in MODELOS))
    registrar("validação: configuração %s, semente %d, 12 dobras de 2023 (treino com todo alvo anterior ao mês, desde "
              "jan/2019; previsão das 166 linhas do mês)" % (c.CONFIG, comum.SEMENTE))
    linhas, previsoes, total = [], [], {}
    for m in MODELOS:
        prev, _ = comum.prever_dobras(base, nomes[m], c.CONFIG)
        por_dobra = []
        for mes, g in prev.groupby("mes_alvo", sort=True):
            assert len(g) == 166
            por_dobra.append({"modelo": m, "atributos": len(nomes[m]), "dobra": "2023-%02d" % mes, "linhas": len(g),
                              **comum.metricas(g[ALVO], g["previsao"])})
        assert len(por_dobra) == 12
        media = {k: float(np.mean([d[k] for d in por_dobra])) for k in METRICAS}
        total[m] = comum.metricas(prev[ALVO], prev["previsao"])
        linhas += por_dobra + [
            {"modelo": m, "atributos": len(nomes[m]), "dobra": "media_das_12_dobras", "linhas": 1992, **media},
            {"modelo": m, "atributos": len(nomes[m]), "dobra": "todas_as_1992_linhas", "linhas": 1992, **total[m]}]
        previsoes.append(prev.rename(columns={ALVO: "alvo"}).assign(modelo=m, bairro=prev["bairro"].astype(str)))
        registrar("validação, %s (%d atributos): nas 1.992 linhas MAE %s, RMSE %s, deviance %s; média das 12 dobras "
                  "MAE %s, RMSE %s, deviance %s" % (m, len(nomes[m]), *map(seis, total[m].values()),
                                                     *map(seis, media.values())))
    c.gravar_csv(pd.DataFrame(linhas), SAIDA[2023])
    c.gravar_csv(pd.concat(previsoes, ignore_index=True)[["modelo", "bairro", "ano_alvo", "mes_alvo", "alvo",
                                                          "previsao"]], PREVISOES[2023])
    registrar("validação: gravados %s e %s" % (SAIDA[2023].name, PREVISOES[2023].name))
    registrar("validação: fim; nenhuma linha de alvo em 2024 foi usada e nenhuma métrica de 2024 foi calculada")


def teste():
    """Corte fixo de 2024 para os 4 modelos e o resumo dos dois anos."""
    registrar("teste: início; leitura da base com as linhas de alvo em 2024")
    base = c.ler_base("teste")
    nomes = conferir_blocos()
    metricas, previsoes = corte_fixo(base, 2024, list(MODELOS), nomes)
    tabela = tabela_do_teste(metricas, nomes)
    c.gravar_csv(tabela, SAIDA[2024])
    c.gravar_csv(previsoes[["modelo", "bairro", "ano_alvo", "mes_alvo", "alvo", "previsao"]], PREVISOES[2024])
    for m in MODELOS:
        registrar("teste, %s (%d atributos), corte fixo, 1.992 linhas de 2024: MAE %s, RMSE %s, deviance %s"
                  % (m, len(nomes[m]), *map(seis, metricas[m].values())))
    registrar("teste: gravados %s e %s" % (SAIDA[2024].name, PREVISOES[2024].name))
    c.gravar_json(resumo_de(pd.read_csv(SAIDA[2023]), tabela, nomes), RESUMO)
    registrar("teste: fim; gravado %s" % RESUMO.name)


if __name__ == "__main__":
    assert len(sys.argv) == 2 and sys.argv[1] in PASSOS, "uso: ablacao_blocos.py validacao|teste"
    RESULTADOS.mkdir(exist_ok=True)
    {"validacao": validacao, "teste": teste}[sys.argv[1]]()
