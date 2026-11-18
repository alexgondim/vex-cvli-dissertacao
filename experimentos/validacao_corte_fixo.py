"""Validação no corte fixo: o modelo treinado com os alvos de 2019 a 2022, sem reajuste, ao lado das quatro
referências de comparação (binomial negativa, persistência, média móvel de 12 meses e média sazonal).

Lê `resultados/contribuicoes_corte_fixo.csv` (as estimativas do modelo) e a base por bairro e mês. Ajusta a binomial
negativa nas 7.968 linhas de alvo de 2019 a 2022. A persistência, a média móvel de 12 meses e a média sazonal usam
só o histórico de cada bairro. Avalia nas 1.992 linhas do ano: 2023 no modo `ensaio`, 2024 no modo `teste`.

Grava `validacao_metricas_corte_fixo.csv`, `validacao_previsoes_corte_fixo.csv` e `validacao_nb_corte_fixo.json`.

Da raiz do repositório: python experimentos/validacao_corte_fixo.py teste
"""
import numpy as np
import pandas as pd

import validacao_comum as c

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
c.gravar_csv(metricas, saida / "validacao_metricas_corte_fixo.csv")
c.gravar_csv(tabela, saida / "validacao_previsoes_corte_fixo.csv")
c.gravar_json(c.resumo_nb(nb, "corte fixo"), saida / "validacao_nb_corte_fixo.json")
c.registrar("%s, corte fixo: fim; gravados validacao_metricas_corte_fixo.csv, validacao_previsoes_corte_fixo.csv e "
            "validacao_nb_corte_fixo.json" % m)
