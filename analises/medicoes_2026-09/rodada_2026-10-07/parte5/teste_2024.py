"""Item 4 do bloco 42, pela rotina do corte fixo (treino nas 7.968 linhas de alvo de 2019 a 2022), como MA, MB e MC
na Parte 2 e a MD na Parte 3.

No teste (1 vez, depois do congelamento), avaliação nas 1.992 linhas de 2024: primeiro a MD tem de repetir as 3
métricas de 2024 já vistas em 13/10 até a 6ª casa; se não repetir, para antes das variantes novas. Depois ME, MF e
MG, só as 3 métricas. Grava conferencia_2024.json e sensibilidade_parte5_2024.csv.

No ensaio (antes do congelamento), o mesmo programa roda avaliado nas 1.992 linhas de 2023, para conferir a rotina
sem olhar 2024: a MD tem de repetir o corte fixo de 2023 da Parte 3 (sensibilidade_MD_2023.csv). Só registra no log.
"""
import sys

import pandas as pd

import comum5 as c

CASAS = 6
REFERENCIA_MD = {"teste": {"mae": 0.570633, "rmse": 0.833318, "deviance": 1.022385},
                 "ensaio": {"mae": 0.590529, "rmse": 0.852938, "deviance": 1.043370}}

m = c.modo()
ano = c.ANO[m]
c.registrar("%s, item 4: início; corte fixo avaliado nas linhas de alvo em %d" % (m, ano))
base = c.c3.ler_base(m)
treino, avaliadas = c.c3.corte_fixo(base, ano)
y = avaliadas[c.ALVO].to_numpy(float)


def corte_fixo(nomes):
    modelo = c.comum.estimador(c.CONFIG).fit(treino[nomes], treino[c.ALVO].to_numpy(float))
    return c.comum.metricas(y, modelo.predict(avaliadas[nomes]).astype(float))


m_md = corte_fixo(c.MD)
ok = all(round(m_md[k], CASAS) == REFERENCIA_MD[m][k] for k in REFERENCIA_MD[m])
c.registrar("%s, item 4, MD (%d atributos) no corte fixo de %d: MAE %.6f, RMSE %.6f, deviance %.6f (%r, %r, %r); "
            "repete %s: %s" % (m, len(c.MD), ano, m_md["mae"], m_md["rmse"], m_md["deviance"], m_md["mae"],
                              m_md["rmse"], m_md["deviance"], REFERENCIA_MD[m], ok))
if m == "teste":
    c.c2.gravar_json({"ok": bool(ok), "criterio": "as 3 métricas da MD no corte fixo de 2024 iguais às vistas em 13/10 "
                      "(parte3/sensibilidade_MD_2024.csv) até a 6ª casa", "ano": ano, "MD": m_md,
                      "esperado": REFERENCIA_MD[m]}, c.PARTE5 / "conferencia_2024.json")
if not ok:
    c.registrar("PARADA: a MD não repete o corte fixo de %d" % ano)
    sys.exit(1)

linhas = []
for v, nomes in c.VARIANTES.items():
    met = corte_fixo(nomes)
    esp = c.espaciais_de(nomes)
    c.registrar("%s, item 4, %s (%d atributos; espaciais: %s) no corte fixo de %d: MAE %.6f, RMSE %.6f, deviance %.6f "
                "(%r, %r, %r)" % (m, v, len(nomes), ", ".join(esp) or "nenhum", ano, met["mae"], met["rmse"],
                                  met["deviance"], met["mae"], met["rmse"], met["deviance"]))
    linhas.append({"variante": v, "atributos": len(nomes), "espaciais": ", ".join(esp) or "nenhum",
                   "forma": "corte_fixo", "ano": ano, "linhas": int(len(y)), **met})
if m == "teste":
    c.c2.gravar_csv(pd.DataFrame(linhas), c.PARTE5 / "sensibilidade_parte5_2024.csv")
    c.registrar("teste, item 4: fim; gravados conferencia_2024.json e sensibilidade_parte5_2024.csv")
else:
    c.registrar("ensaio, item 4: fim; nada gravado além do log")
