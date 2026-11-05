"""Forma (iii), ablação (M1 a M4) e sensibilidade territorial (MA, MB e MC) no corte fixo: cada variante treinada
nas 7.968 linhas de alvo de 2019 a 2022, com a configuração 3, e avaliada nas 1.992 linhas do ano (2023 no ensaio,
2024 no teste). Só as 3 métricas, sem contribuições.

Grava ablacao_sensibilidade_<ano>.csv e previsoes_ablacao_sensibilidade_<ano>.csv.
"""
import pandas as pd

import comum2 as c

B = c.comum.BLOCOS
NAO_ESPACIAIS = B["criminal"] + B["temporal"] + B["perfil"]
VARIANTES = {"M1": B["criminal"], "M2": B["criminal"] + B["temporal"],
             "M3": B["criminal"] + B["temporal"] + B["espacial"], "M4": sum(B.values(), []),
             "MA": NAO_ESPACIAIS + ["bairro"], "MB": NAO_ESPACIAIS + ["risp"], "MC": NAO_ESPACIAIS + ["bairro", "risp"]}
QUANTOS = {"M1": 16, "M2": 20, "M3": 23, "M4": 30, "MA": 28, "MB": 28, "MC": 29}

m = c.modo()
ano = c.ANO[m]
saida = c.pasta_de_saida(m)
c.registrar("%s, ablação e sensibilidade: início, avaliação nas linhas de alvo em %d" % (m, ano))
base = c.ler_base(m)
treino = base[base["ano_alvo"].between(*c.ANOS_DE_TREINO)]
avaliadas = base[base["ano_alvo"] == ano]
assert len(treino) == 7968 and len(avaliadas) == 1992
y = avaliadas[c.ALVO].to_numpy(float)
chave = ["bairro", "ano_alvo", "mes_alvo"]
linhas = avaliadas[chave].assign(bairro=avaliadas["bairro"].astype(str))
metricas, partes = [], []
for variante, atributos in VARIANTES.items():
    nomes = c.comum.em_ordem(atributos)
    assert len(nomes) == QUANTOS[variante], variante
    modelo = c.comum.estimador(c.CONFIG).fit(treino[nomes], treino[c.ALVO].to_numpy(float))
    q = modelo.predict(avaliadas[nomes]).astype(float)
    metricas.append({"variante": variante, "atributos": len(nomes),
                     **c.linha_de_metricas(y, q, forma="corte_fixo", ano=ano)})
    partes.append(linhas.assign(variante=variante, alvo=y, previsao=q))
c.gravar_csv(pd.DataFrame(metricas), saida / ("ablacao_sensibilidade_%d.csv" % ano))
c.gravar_csv(pd.concat(partes, ignore_index=True), saida / ("previsoes_ablacao_sensibilidade_%d.csv" % ano))
c.registrar("%s, ablação e sensibilidade: fim; gravados ablacao_sensibilidade_%d.csv e "
            "previsoes_ablacao_sensibilidade_%d.csv" % (m, ano, ano))
