"""Treina o modelo, uma vez, com a configuração escolhida, nas 7.968 linhas de alvo de 2019 a 2022, e calcula as
contribuições de cada atributo nas 11.952 linhas.

Lê `resultados/escolha.json` e a base por bairro e mês. As contribuições (pred_contribs) ficam na escala do log da
contagem esperada; o programa imprime a maior diferença da aditividade (soma das 30 contribuições mais o valor de
base, contra a margem). Calcula também as interações (pred_interactions) nas 1.992 linhas de 2024, das 3 proporções
de turno e da tendência com bairro, aisp e risp; a interação(i, j) é o elemento [i, j] da matriz de cada linha, sem
dobrar.

Grava em `resultados/`: `modelo_treinado.json`; `estimativas_corte_fixo.csv`, com a margem e a estimativa das 11.952
linhas e a marca de treino, validação ou teste, sem o alvo; `contribuicoes_corte_fixo.csv`; e `interacoes_2024.csv`.

Da raiz do repositório: python modelo/treinar_modelo.py
"""
import json

import numpy as np
import pandas as pd
import xgboost as xgb

import comum

DEPENDENCIA = ["prop_madrugada_hist12", "prop_tarde_hist12", "prop_noite_hist12", "cvli_trend"]
config = json.loads(comum.ESCOLHA.read_text(encoding="utf-8"))["configuracao"]
base = comum.ler_base()
nomes = comum.atributos()
treino = base["ano_alvo"].between(2019, 2022)
assert treino.sum() == 7968

comum.registrar("modelo do corte fixo: treino com %d linhas, configuração %s" % (treino.sum(), config))
modelo = comum.estimador(config).fit(base.loc[treino, nomes], base.loc[treino, comum.ALVO].to_numpy(float))
modelo.save_model(comum.PASTA / "modelo_treinado.json")
X = base[nomes]
identificacao = base[["bairro", "ano_alvo", "mes_alvo", "divisao"]].reset_index(drop=True)
margem = modelo.predict(X, output_margin=True)
estimativa = modelo.predict(X)
estimativas = identificacao.assign(margem=margem, estimativa=estimativa)
estimativas.to_csv(comum.PASTA / "estimativas_corte_fixo.csv", index=False)
comum.registrar("modelo gravado (modelo_treinado.json) e estimativas das %d linhas: %s"
                % (len(estimativas), estimativas["divisao"].value_counts().to_dict()))

comum.registrar("contribuições do corte fixo")
contrib = comum.contribuicoes(modelo, X)
desvio = np.abs(contrib.sum(axis=1) - margem)
comum.registrar("aditividade nas %d linhas: maior |soma das 30 contribuições + valor de base - margem| = %.3g"
                % (len(desvio), desvio.max()))
tabela = identificacao.assign(valor_base=contrib[:, -1])
for k, a in enumerate(nomes):
    tabela["contrib_" + a] = contrib[:, k]
tabela = tabela.assign(margem=margem, estimativa=estimativa)
tabela.to_csv(comum.PASTA / "contribuicoes_corte_fixo.csv", index=False)

linhas_2024 = (base["ano_alvo"] == 2024).to_numpy()
inter = modelo.get_booster().predict(xgb.DMatrix(X[linhas_2024], enable_categorical=True), pred_interactions=True)
assert inter.shape == (1992, 31, 31)
linha_contra_contrib = np.abs(inter.sum(axis=2) - contrib[linhas_2024]).max()
comum.registrar("interações nas 1.992 linhas de 2024: maior |soma da linha i da matriz - contribuição de i| = %.3g"
                % linha_contra_contrib)
saida = identificacao[linhas_2024].reset_index(drop=True).assign(risp=base.loc[linhas_2024, "risp"].astype(str).to_numpy())
for a in DEPENDENCIA:
    i = nomes.index(a)
    for e in comum.BLOCOS["espacial"]:
        saida["%s__%s" % (a, e)] = inter[:, i, nomes.index(e)]
    saida[a + "__soma"] = sum(saida["%s__%s" % (a, e)] for e in comum.BLOCOS["espacial"])
    comum.registrar("%s: média do valor absoluto da soma das interações com bairro, aisp e risp = %.6f"
                    % (a, saida[a + "__soma"].abs().mean()))
saida.to_csv(comum.PASTA / "interacoes_2024.csv", index=False)
comum.registrar("etapas 7 e 8: fim")
