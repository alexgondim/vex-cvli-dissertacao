"""Variantes ME (bairro e aisp), MF (só aisp) e MG (nenhum atributo espacial) no corte fixo: treino nas 7.968
linhas de alvo de 2019 a 2022 e avaliação nas 1.992 linhas do ano, só as 3 métricas.

No modo `teste`, avalia em 2024 e grava `resultados/sensibilidade_variantes_corte_fixo_2024.csv`. No modo `ensaio`,
avalia em 2023 e só imprime as métricas. Lê a base por bairro e mês.

Da raiz do repositório: python experimentos/sensibilidade_variantes_corte_fixo.py teste
"""

import pandas as pd

import sensibilidade_comum_variantes as c

m = c.modo()
ano = c.ANO[m]
c.registrar("%s: início; corte fixo avaliado nas linhas de alvo em %d" % (m, ano))
base = c.c3.ler_base(m)
treino, avaliadas = c.c3.corte_fixo(base, ano)
y = avaliadas[c.ALVO].to_numpy(float)


def corte_fixo(nomes):
    modelo = c.comum.estimador(c.CONFIG).fit(treino[nomes], treino[c.ALVO].to_numpy(float))
    return c.comum.metricas(y, modelo.predict(avaliadas[nomes]).astype(float))


linhas = []
for v, nomes in c.VARIANTES.items():
    met = corte_fixo(nomes)
    esp = c.espaciais_de(nomes)
    c.registrar("%s, %s (%d atributos; espaciais: %s) no corte fixo de %d: MAE %.6f, RMSE %.6f, deviance %.6f "
                "(%r, %r, %r)" % (m, v, len(nomes), ", ".join(esp) or "nenhum", ano, met["mae"], met["rmse"],
                                  met["deviance"], met["mae"], met["rmse"], met["deviance"]))
    linhas.append({"variante": v, "atributos": len(nomes), "espaciais": ", ".join(esp) or "nenhum",
                   "forma": "corte_fixo", "ano": ano, "linhas": int(len(y)), **met})
if m == "teste":
    c.c2.gravar_csv(pd.DataFrame(linhas), c.RESULTADOS / "sensibilidade_variantes_corte_fixo_2024.csv")
    c.registrar("teste: fim; gravado sensibilidade_variantes_corte_fixo_2024.csv")
else:
    c.registrar("ensaio: fim; nada gravado")
