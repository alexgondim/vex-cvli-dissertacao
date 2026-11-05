"""Itens 8 e 9 do bloco 26: modelo do corte fixo e as contribuições dele.

Treina 1 vez, com a configuração escolhida, nas 7.968 linhas de alvo de 2019 a 2022. Grava o modelo e as
estimativas das 11.952 linhas, com a marca de treino, validação ou teste; o alvo não entra nesses arquivos.
Contribuições (pred_contribs) nas 11.952 linhas, na escala do log da contagem esperada, com a conferência da
aditividade. Interações (pred_interactions) nas 1.992 linhas de alvo em 2024: para as 3 proporções de turno e a
tendência, a interação com bairro, com aisp e com risp e a soma das três. Como no bloco 9, a interação(i, j) é o
elemento [i, j] da matriz de cada linha, sem dobrar.
"""
import json

import numpy as np
import pandas as pd
import xgboost as xgb

import comum

assert (comum.PASTA / "CONGELADO.tsv").exists(), "o modelo do corte fixo vem depois do congelamento"
DEPENDENCIA = ["prop_madrugada_hist12", "prop_tarde_hist12", "prop_noite_hist12", "cvli_trend"]
config = json.loads(comum.ESCOLHA.read_text(encoding="utf-8"))["configuracao"]
base = comum.ler_base()
nomes = comum.atributos()
treino = base["ano_alvo"].between(2019, 2022)
assert treino.sum() == 7968

comum.registrar("etapa 7, modelo do corte fixo: treino com %d linhas, configuração %s" % (treino.sum(), config))
modelo = comum.estimador(config).fit(base.loc[treino, nomes], base.loc[treino, comum.ALVO].to_numpy(float))
modelo.save_model(comum.PASTA / "modelo_corte_fixo.json")
X = base[nomes]
identificacao = base[["bairro", "ano_alvo", "mes_alvo", "divisao"]].reset_index(drop=True)
margem = modelo.predict(X, output_margin=True)
estimativa = modelo.predict(X)
estimativas = identificacao.assign(margem=margem, estimativa=estimativa)
estimativas.to_csv(comum.PASTA / "estimativas_corte_fixo.csv", index=False)
comum.registrar("modelo gravado (modelo_corte_fixo.json) e estimativas das %d linhas: %s"
                % (len(estimativas), estimativas["divisao"].value_counts().to_dict()))

comum.registrar("etapa 8, contribuições do corte fixo")
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
