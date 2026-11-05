"""Forma (i), corte fixo. O modelo é o do corte fixo de 07/10, sem reajuste, com as estimativas gravadas em
contribuicoes_corte_fixo.csv; a binomial negativa é ajustada nas 7.968 linhas de alvo de 2019 a 2022; as
referências ingênuas usam só o histórico de cada bairro. Tudo avaliado nas 1.992 linhas do ano (2023 no ensaio,
2024 no teste).

Grava metricas_corte_fixo.csv, previsoes_corte_fixo.csv e nb_corte_fixo.json.
"""
import numpy as np
import pandas as pd

import comum2 as c

m = c.modo()
ano = c.ANO[m]
saida = c.pasta_de_saida(m)
c.registrar("%s, corte fixo: início, avaliação nas linhas de alvo em %d" % (m, ano))
base = c.ler_base(m)
treino = base[base["ano_alvo"].between(*c.ANOS_DE_TREINO)]
avaliadas = base[base["ano_alvo"] == ano]
assert len(treino) == 7968 and len(avaliadas) == 1992

chave = ["bairro", "ano_alvo", "mes_alvo"]
estimativas = pd.read_csv(c.ESTIMATIVAS, usecols=chave + ["estimativa"])
estimativas = estimativas[estimativas["ano_alvo"] == ano]
linhas = avaliadas[chave + [c.ALVO]].assign(bairro=avaliadas["bairro"].astype(str))
linhas = linhas.merge(estimativas, on=chave, how="left", validate="one_to_one")
assert len(linhas) == 1992 and linhas["estimativa"].notna().all()

nb = c.ajustar_nb(treino)
numericas = treino[c.NB_NUMERICAS].to_numpy(float)
X = c.desenho_nb(treino, (numericas.mean(axis=0), numericas.std(axis=0)), nb["aisps"])
inicio = np.zeros(X.shape[1] + 1)
inicio[0] = np.log(treino[c.ALVO].mean())
erro = c.conferir_gradiente(X, treino[c.ALVO].to_numpy(float), inicio)
assert erro < 1e-6, erro
c.registrar("%s, corte fixo: binomial negativa convergiu %s (iterações %d, recomeços %d, maior |gradiente| %.1e); "
            "gradiente analítico conferido, erro relativo %.1e" % (m, nb["convergiu"], nb["iteracoes"], nb["recomecos"],
                                                                    nb["gradiente_max"], erro))

referencias = c.referencias(base).loc[avaliadas.index]
previsoes = {"modelo": linhas["estimativa"].to_numpy(float), "binomial_negativa": c.prever_nb(nb, avaliadas),
             "persistencia": referencias["persistencia"].to_numpy(), "media_movel_12": referencias["media_movel_12"].to_numpy(),
             "media_sazonal": referencias["media_sazonal"].to_numpy()}
y = linhas[c.ALVO].to_numpy(float)
metricas = pd.DataFrame([c.linha_de_metricas(y, previsoes[k], forma="corte_fixo", ano=ano, metodo=k) for k in c.METODOS])
tabela = pd.concat([linhas[chave].assign(forma="corte_fixo", metodo=k, alvo=y, previsao=previsoes[k])
                    for k in c.METODOS], ignore_index=True)
c.gravar_csv(metricas, saida / "metricas_corte_fixo.csv")
c.gravar_csv(tabela, saida / "previsoes_corte_fixo.csv")
c.gravar_json(c.resumo_nb(nb, "corte fixo"), saida / "nb_corte_fixo.json")
c.registrar("%s, corte fixo: fim; gravados metricas_corte_fixo.csv, previsoes_corte_fixo.csv e nb_corte_fixo.json"
            % m)
