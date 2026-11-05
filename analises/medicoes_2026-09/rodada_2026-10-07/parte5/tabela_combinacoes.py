"""Item 6 do bloco 42: a tabela das 8 combinações de atributos espaciais, M4, MA, MB, MC e MD da rodada e ME, MF e MG
desta parte. Para cada uma: atributos espaciais, deviance nas 12 dobras de 2023, deviance no corte fixo de 2024,
peso espacial em 2023, atributo de maior média do valor absoluto em 2023 e tau-b dos 27 não espaciais contra o M4.
Só lê arquivos já gravados: não ajusta modelo nem calcula métrica.

Fontes:
- deviance de 2023: sensibilidade_metricas.csv (M4, MA, MB e MC), parte3/sensibilidade_MD_2023.csv (linha das
  dobras) e parte5/sensibilidade_parte5_2023.csv;
- deviance de 2024: parte2/ablacao_sensibilidade_2024.csv (M4, MA, MB e MC, a mesma rotina do corte fixo),
  parte3/sensibilidade_MD_2024.csv e parte5/sensibilidade_parte5_2024.csv;
- peso espacial e maior média de 2023: M4, MA, MB e MC calculados aqui das contribuições da Parte 1
  (contribuicoes_sensibilidade.csv), com as funções do comum3.py, como o md_validacao.py fez para o M4 e a MB; MD de
  parte3/resumo_MD_2023.json; ME, MF e MG de parte5/resumo_parte5.json;
- tau-b contra o M4: sensibilidade_tau_b.csv (MA, MB e MC), parte3/resumo_MD_2023.json e parte5/resumo_parte5.json;
  no M4, 1 por definição.

No ensaio (antes do congelamento), monta a tabela sem nenhuma coluna de 2024 e só registra no log; no teste, com as
deviances de 2024, grava tabela_8_combinacoes.csv.
"""
import json

import numpy as np
import pandas as pd

import comum5 as c

m = c.modo()
c.registrar("%s, item 6: início; tabela das 8 combinações, só com arquivos já gravados" % m)
R, P3, P5 = c.RODADA, c.RODADA / "parte3", c.PARTE5
NOMES = {"M4": c.comum.atributos(), "MA": c.MA, "MB": c.c3.MB,
         "MC": c.comum.em_ordem(c.NAO_ESPACIAIS + ["bairro", "risp"]), "MD": c.MD, **c.VARIANTES}
assert [len(v) for v in NOMES.values()] == [30, 28, 28, 29, 29, 29, 28, 27]
ORIGEM = {"M4": "Parte 1", "MA": "Parte 1", "MB": "Parte 1", "MC": "Parte 1", "MD": "Parte 3",
          "ME": "Parte 5", "MF": "Parte 5", "MG": "Parte 5"}


def metricas_de(arquivo, coluna, filtro=None):
    """As 3 métricas de cada variante num CSV de métricas, pela coluna que dá o nome da variante. O leitor padrão do
    pandas pode devolver o número do texto com 1 unidade de arredondamento a menos ou a mais (a deviance da MD em
    parte3/sensibilidade_MD_2023.csv sai 2,2e-16 abaixo); round_trip devolve o float exato que foi gravado."""
    t = pd.read_csv(arquivo, float_precision="round_trip")
    if filtro:
        t = t[t[filtro[0]] == filtro[1]]
    assert t[coluna].is_unique
    return t.set_index(coluna)[["mae", "rmse", "deviance"]].to_dict("index")


def deviance_de(arquivo, coluna, filtro=None):
    return {n: v["deviance"] for n, v in metricas_de(arquivo, coluna, filtro).items()}


dev23 = {**deviance_de(R / "sensibilidade_metricas.csv", "modelo"),
         **deviance_de(P3 / "sensibilidade_MD_2023.csv", "variante", ("forma", "dobras_2023")),
         **deviance_de(P5 / "sensibilidade_parte5_2023.csv", "variante")}
dev24 = {}
if m == "teste":
    p2 = deviance_de(R / "parte2" / "ablacao_sensibilidade_2024.csv", "variante")
    dev24 = {**{k: p2[k] for k in ("M4", "MA", "MB", "MC")},
             **deviance_de(P3 / "sensibilidade_MD_2024.csv", "variante"),
             **deviance_de(P5 / "sensibilidade_parte5_2024.csv", "variante")}

# a MF (os 27 e aisp) contra a MD (os 27, aisp e risp): na MD, a risp não entra em nenhum corte
md23 = pd.read_csv(P3 / "contribuicoes_MD_2023.csv", float_precision="round_trip")
mf23 = pd.read_csv(P5 / "contribuicoes_MF_2023.csv", float_precision="round_trip")
colunas = ["contrib_" + a for a in NOMES["MF"]] + ["valor_base"]
chave = ["bairro", "ano_alvo", "mes_alvo"]
iguais23 = bool((md23[chave].to_numpy() == mf23[chave].to_numpy()).all() and np.array_equal(
    md23[colunas].to_numpy().astype(np.float32), mf23[colunas].to_numpy().astype(np.float32)))
c.registrar("%s, item 6: na MD, contribuição da risp igual a 0 nas 1.992 linhas de 2023: %s; contribuições da MF "
            "iguais bit a bit (float32) às da MD sem a risp: %s; 3 métricas de 2023 da MF iguais às da MD: %s"
            % (m, bool((md23["contrib_risp"] == 0).all()), iguais23,
               metricas_de(P5 / "sensibilidade_parte5_2023.csv", "variante")["MF"]
               == metricas_de(P3 / "sensibilidade_MD_2023.csv", "variante", ("forma", "dobras_2023"))["MD"]))
if m == "teste":
    c.registrar("teste, item 6: 3 métricas de 2024 da MF iguais às da MD: %s"
                % (metricas_de(P5 / "sensibilidade_parte5_2024.csv", "variante")["MF"]
                   == metricas_de(P3 / "sensibilidade_MD_2024.csv", "variante")["MD"]))

# peso espacial e maior média da Parte 1, das contribuições gravadas lá; com o leitor padrão do pandas, o mesmo do
# md_validacao.py, para o M4 e a MB repetirem o peso de parte3/resumo_MD_2023.json
cs = pd.read_csv(R / "contribuicoes_sensibilidade.csv")
peso, maior = {}, {}
for n in ("M4", "MA", "MB", "MC"):
    d = cs[cs["modelo"] == n][["contrib_" + a for a in NOMES[n]]]
    assert len(d) == 1992
    d.columns = NOMES[n]
    medias = c.c3.media_abs(d, NOMES[n])
    peso[n] = c.c3.peso_espacial(medias, c.ESPACIAIS)
    o = medias.sort_values(ascending=False)
    maior[n] = (o.index[0], float(o.iloc[0]))
resumo_md = json.loads((P3 / "resumo_MD_2023.json").read_text(encoding="utf-8"))
# o M4 e a MB já tinham o peso espacial, pela mesma conta, no resumo da MD
repete_p3 = {n: peso[n] == resumo_md["peso_espacial"][n] for n in ("M4", "MB")}
c.registrar("%s, item 6: peso espacial do M4 e da MB, recalculado das contribuições da Parte 1, igual ao de "
            "parte3/resumo_MD_2023.json: %s" % (m, repete_p3))
peso["MD"] = resumo_md["peso_espacial"]["MD"]
maior["MD"] = (resumo_md["ordem_29_MD"][0], resumo_md["media_abs_29"][resumo_md["ordem_29_MD"][0]])
resumo5 = json.loads((P5 / "resumo_parte5.json").read_text(encoding="utf-8"))["variantes"]
for v in c.VARIANTES:
    peso[v] = resumo5[v]["peso_espacial"]
    maior[v] = (resumo5[v]["maior_media_abs"]["atributo"], resumo5[v]["maior_media_abs"]["valor"])

taus = pd.read_csv(R / "sensibilidade_tau_b.csv").set_index("par")["tau_b"]
tau = {"M4": 1.0, **{n: float(taus["M4 x " + n]) for n in ("MA", "MB", "MC")},
       "MD": resumo_md["tau_b"]["MD x M4"], **{v: resumo5[v]["tau_b_contra_M4"] for v in c.VARIANTES}}

tabela = pd.DataFrame([{"combinacao": n, "origem": ORIGEM[n], "atributos": len(NOMES[n]),
                        "espaciais": ", ".join(c.espaciais_de(NOMES[n])) or "nenhum",
                        "deviance_2023_dobras": dev23[n], "deviance_2024_corte_fixo": dev24.get(n, np.nan),
                        "peso_espacial_2023": peso[n], "maior_media_abs_2023": maior[n][0],
                        "valor_maior_media_abs_2023": maior[n][1], "tau_b_contra_M4": tau[n]} for n in NOMES])
assert tabela["deviance_2023_dobras"].notna().all()
assert m == "ensaio" or tabela["deviance_2024_corte_fixo"].notna().all()
for _, t in tabela.iterrows():
    c.registrar("%s, item 6, %s (%s; espaciais: %s): deviance 2023 %.6f; deviance 2024 %s; peso espacial 2023 %.6f; "
                "maior média 2023 %s (%.6f); tau-b contra o M4 %.6f"
                % (m, t["combinacao"], t["origem"], t["espaciais"], t["deviance_2023_dobras"],
                   "%.6f" % t["deviance_2024_corte_fixo"] if m == "teste" else "fora do ensaio",
                   t["peso_espacial_2023"], t["maior_media_abs_2023"], t["valor_maior_media_abs_2023"],
                   t["tau_b_contra_M4"]))
if m == "teste":
    c.c2.gravar_csv(tabela, P5 / "tabela_8_combinacoes.csv")
    c.registrar("teste, item 6: fim; gravado tabela_8_combinacoes.csv")
else:
    c.registrar("ensaio, item 6: fim; sem as colunas de 2024; nada gravado além do log")
