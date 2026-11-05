"""Bloco 37, Parte 4 da rodada de 07/10: ablação por blocos com a composição pedida pelo orientador em 21/10.

M1 criminal, M2 + temporal, M3 + espacial e M4 com os 30 atributos, todos com a configuração 3 da grade e a semente
42, sem busca. Validação nas 12 dobras de 2023 da janela expansiva, as da Parte 1; teste no corte fixo, com treino
nas 7.968 linhas de alvo de 2019 a 2022 e avaliação nas 1.992 de 2024, 1 vez por modelo, só depois do congelamento.

Reaproveita por import o comum2.py da parte2/ e, por ele, o comum.py desta pasta: base de 27/09 conferida pelo
sha256, os 30 atributos na ordem do dicionário, configuração 3, estimador, métricas (piso de 0,01 só na deviance),
dobras de 2023 e previsão por dobra. Do ablacao_sensibilidade.py não importa nada: ele não tem funções, e importá-lo
rodaria de novo a ablação da Parte 1 e regravaria as saídas dela. Não grava nada nas pastas das Partes 1 a 3, nem
bytecode; log, conferência, saídas e congelamento ficam em parte4/, sem sobrescrever.

Da raiz do repositório, nesta ordem:
PYTHONUTF8=1 ./venv/Scripts/python.exe analises/medicoes_2026-09/rodada_2026-10-07/ablacao_blocos2.py validacao
PYTHONUTF8=1 ./venv/Scripts/python.exe analises/medicoes_2026-09/rodada_2026-10-07/ablacao_blocos2.py congelar
PYTHONUTF8=1 ./venv/Scripts/python.exe analises/medicoes_2026-09/rodada_2026-10-07/ablacao_blocos2.py teste
"""
import sys

sys.dont_write_bytecode = True  # o import do comum2 e do comum não pode deixar bytecode nas pastas das Partes 1 e 2

import datetime  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RODADA = Path(__file__).resolve().parent
sys.path.insert(0, str(RODADA / "parte2"))
import comum2 as c  # noqa: E402

comum = c.comum
ALVO = comum.ALVO
PARTE4 = RODADA / "parte4"
LOG = PARTE4 / "rodada_parte4.log"
PROTOCOLO = PARTE4 / "protocolo_parte4.md"
CONGELADO = PARTE4 / "CONGELADO_parte4.tsv"
CONFERENCIA_2023 = PARTE4 / "conferencia_2023.json"
SAIDA = {2023: PARTE4 / "ablacao_blocos2_2023.csv", 2024: PARTE4 / "ablacao_blocos2_2024.csv"}
PREVISOES = {2023: PARTE4 / "previsoes_ablacao_blocos2_2023.csv", 2024: PARTE4 / "previsoes_ablacao_blocos2_2024.csv"}
RESUMO = PARTE4 / "resumo_parte4.json"
# caminhos a partir desta pasta, como ficam no CONGELADO_parte4.tsv
CONGELAR = ["parte4/protocolo_parte4.md", "ablacao_blocos2.py", "parte2/comum2.py"]
SHA_COMUM_PARTE1 = "f121fa82c4957c4a7b27f827450fdb0949e8044a3c5fa892d2fe03387a6aa2d3"  # CONGELADO.tsv da Parte 1

# Composição dos blocos pedida pelo orientador em 21/10. Substitui, neste bloco e no capítulo, a da Parte 1
# (comum.BLOCOS), que fica como está nos programas dela: as 4 proporções de turno e de fim de semana saem do perfil
# e entram no temporal; criminal e espacial não mudam.
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

# M1 e M4 têm os mesmos atributos da Parte 1 e têm de repetir até a 6ª casa os valores que ela e a Parte 2 gravaram
REFERENCIA = {"validação": (RODADA / "ablacao.csv", "modelo"),
              "ensaio do teste": (RODADA / "parte2" / "ensaio" / "ablacao_sensibilidade_2023.csv", "variante"),
              "teste": (RODADA / "parte2" / "ablacao_sensibilidade_2024.csv", "variante")}
METRICAS = ("mae", "rmse", "deviance")
PASSOS = ("validacao", "congelar", "teste")


def agora():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def registrar(texto):
    linha = "%s  %s" % (agora(), texto)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(linha + "\n")
    print(linha, flush=True)


def seis(v):
    return "%.6f" % v


def conferir_blocos():
    """Item 1, por código, antes de treinar: blocos sem atributo repetido, com 16, 8, 3 e 3 atributos, que juntos são
    os 30 do dicionário; criminal e espacial iguais aos da Parte 1; temporal = o da Parte 1 mais as 4 proporções, e
    perfil = o da Parte 1 menos elas. Devolve os atributos de cada modelo, na ordem do dicionário."""
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


def conferir(etapa, metricas):
    """M1 e M4 contra o valor gravado antes, escrito com 6 casas. Devolve se repetiu e as linhas da conferência."""
    arquivo, coluna = REFERENCIA[etapa]
    ref = pd.read_csv(arquivo).set_index(coluna)
    linhas, ok = [], True
    for m in ("M1", "M4"):
        for k in METRICAS:
            igual = seis(metricas[m][k]) == seis(ref.loc[m, k])
            ok &= igual
            linhas.append({"etapa": etapa, "modelo": m, "metrica": k, "agora": metricas[m][k],
                           "gravado": float(ref.loc[m, k]), "arquivo": arquivo.relative_to(RODADA).as_posix(),
                           "igual_na_6a_casa": igual})
    registrar("%s, conferência de M1 e M4 contra %s, 6 casas: %s" % (
        etapa, arquivo.relative_to(RODADA).as_posix(),
        "; ".join("%s %s %s contra %s" % (l["modelo"], l["metrica"], seis(l["agora"]), seis(l["gravado"]))
                  for l in linhas)))
    return ok, linhas


def corte_fixo(base, ano, modelos, nomes):
    """Como a forma (iii) da Parte 2 (parte2/forma_ablacao.py): cada modelo treinado nas 7.968 linhas de alvo de 2019
    a 2022 e avaliado nas 1.992 linhas do ano."""
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
    """Item 7: os 4 modelos lado a lado em 2023 (média das 12 dobras e 1.992 linhas) e em 2024, e o ganho de cada
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
    """Item 4: as 12 dobras de 2023 para os 4 modelos, a conferência de M1 e M4 com a Parte 1 e o ensaio do caminho
    do teste com 2023 no lugar de 2024 (só M1 e M4, contra o ensaio da Parte 2)."""
    assert PROTOCOLO.exists(), "o protocolo vem antes da validação"
    assert not CONGELADO.exists(), "a validação é anterior ao congelamento"
    registrar("validação: início; protocolo_parte4.md presente, sha256 %s" % c.sha256(PROTOCOLO))
    base = c.ler_base("ensaio")
    assert base["ano_alvo"].max() == 2023
    registrar("validação: base de 27/09 conferida pelo sha256 (%s); as linhas de alvo em 2024 saíram logo depois da "
              "leitura; ficam %d linhas, com alvo de %d a %d" % (comum.SHA_BASE, len(base), base["ano_alvo"].min(),
                                                                base["ano_alvo"].max()))
    nomes = conferir_blocos()
    registrar("item 1 conferido por código: criminal 16, temporal 8, espacial 3, perfil 3, sem repetição, somando os "
              "30 do dicionário; M1 %d, M2 %d, M3 %d, M4 %d atributos; M1 e M4 iguais aos da Parte 1"
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
    ok_validacao, conf = conferir("validação", total)
    # o ensaio usa a mesma função do teste, com 2023 avaliado; os números ficam só no log
    ensaio, _ = corte_fixo(base, 2023, ["M1", "M4"], nomes)
    ok_ensaio, conf_ensaio = conferir("ensaio do teste", ensaio)
    c.gravar_csv(pd.DataFrame(linhas), SAIDA[2023])
    c.gravar_csv(pd.concat(previsoes, ignore_index=True)[["modelo", "bairro", "ano_alvo", "mes_alvo", "alvo",
                                                          "previsao"]], PREVISOES[2023])
    c.gravar_json({"ok": bool(ok_validacao and ok_ensaio), "validacao_repetiu": bool(ok_validacao),
                   "ensaio_do_teste_repetiu": bool(ok_ensaio), "conferencias": conf + conf_ensaio,
                   "data_hora": agora()}, CONFERENCIA_2023)
    registrar("validação: gravados %s, %s e %s" % (SAIDA[2023].name, PREVISOES[2023].name, CONFERENCIA_2023.name))
    if not ok_validacao:
        registrar("PARADA: M1 ou M4 não repete na 6ª casa os valores da Parte 1 nas 12 dobras de 2023")
        sys.exit(1)
    if not ok_ensaio:
        registrar("PARADA: o caminho do teste, com 2023 avaliado, não repete o ensaio da Parte 2 em M1 ou M4")
        sys.exit(1)
    registrar("validação: fim; M1 e M4 repetem a Parte 1 e o ensaio da Parte 2 na 6ª casa; nenhuma linha de alvo em "
              "2024 foi usada e nenhuma métrica de 2024 foi calculada")


def congelar():
    """Item 5: CONGELADO_parte4.tsv com o sha256 do protocolo, deste programa e do comum2.py, depois da validação
    conferida."""
    assert PROTOCOLO.exists() and not CONGELADO.exists()
    assert json.loads(CONFERENCIA_2023.read_text(encoding="utf-8"))["ok"], "a validação não fechou"
    assert SAIDA[2023].exists() and not SAIDA[2024].exists()
    quando = agora()
    tabela = pd.DataFrame({"arquivo": CONGELAR, "sha256": [c.sha256(RODADA / a) for a in CONGELAR],
                           "data_hora": quando})
    with open(CONGELADO, "x", encoding="utf-8", newline="") as f:
        tabela.to_csv(f, sep="\t", index=False)
    registrar("congelamento: CONGELADO_parte4.tsv gravado em %s (sha256 %s): %s; até aqui nenhuma linha de alvo em "
              "2024 foi usada e nenhuma métrica de 2024 foi calculada"
              % (quando, c.sha256(CONGELADO), "; ".join("%s %s" % r for r in zip(CONGELAR, tabela["sha256"]))))


def teste():
    """Item 6: 1 vez por modelo, depois do congelamento. As saídas de 2024 são gravadas antes da conferência, para
    que nada precise rodar de novo."""
    assert CONGELADO.exists(), "o teste só roda depois do congelamento"
    for _, l in pd.read_csv(CONGELADO, sep="\t").iterrows():
        assert c.sha256(RODADA / l["arquivo"]) == l["sha256"], "%s mudou depois do congelamento" % l["arquivo"]
    assert c.sha256(RODADA / "comum.py") == SHA_COMUM_PARTE1, "comum.py diferente do congelado na Parte 1"
    assert json.loads(CONFERENCIA_2023.read_text(encoding="utf-8"))["ok"]
    assert not SAIDA[2024].exists() and not PREVISOES[2024].exists() and not RESUMO.exists(), "o teste roda 1 vez"
    registrar("teste: início; os 3 arquivos do CONGELADO_parte4.tsv e o comum.py da Parte 1 iguais aos congelados; "
              "primeira leitura da base com as linhas de alvo em 2024")
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
    ok, _ = conferir("teste", metricas)
    if not ok:
        registrar("PARADA: M1 ou M4 não repete na 6ª casa os valores da Parte 2 em 2024; sem resumo e sem rodar "
                  "de novo")
        sys.exit(1)
    c.gravar_json(resumo_de(pd.read_csv(SAIDA[2023]), tabela, nomes), RESUMO)
    registrar("teste: fim; M1 e M4 repetem a Parte 2 na 6ª casa; gravado %s" % RESUMO.name)


if __name__ == "__main__":
    assert len(sys.argv) == 2 and sys.argv[1] in PASSOS, "uso: ablacao_blocos2.py validacao|congelar|teste"
    PARTE4.mkdir(exist_ok=True)
    {"validacao": validacao, "congelar": congelar, "teste": teste}[sys.argv[1]]()
