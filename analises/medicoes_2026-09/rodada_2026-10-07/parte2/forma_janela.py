"""Forma (ii), janela expansiva: um reajuste por mês, de jan/2022 ao fim da janela (dez/2023 no ensaio, dez/2024
no teste). Em cada mês, o modelo (configuração 3, os 30 atributos) e a binomial negativa são treinados com todas as
linhas de alvo anterior ao mês e preveem as 166 linhas do mês; as referências ingênuas, nos mesmos meses.

Grava metricas_mensais.csv (uma linha por mês e método), metricas_janela.csv (por ano), previsoes_janela.csv e
nb_janela.json.
"""
import numpy as np
import pandas as pd

import comum2 as c

m = c.modo()
saida = c.pasta_de_saida(m)
meses = c.JANELA[m]
c.registrar("%s, janela: início, %d reajustes, de %s a %s" % (m, len(meses), meses[0], meses[-1]))
base = c.ler_base(m)
nomes = c.comum.atributos()
referencias = c.referencias(base)
chave = ["bairro", "ano_alvo", "mes_alvo"]
partes, mensais, ajustes = [], [], []
for i, mes in enumerate(meses):
    treino = base[base["periodo_alvo"] < mes]
    teste = base[base["periodo_alvo"] == mes]
    assert len(teste) == 166
    modelo = c.comum.estimador(c.CONFIG).fit(treino[nomes], treino[c.ALVO].to_numpy(float))
    nb = c.ajustar_nb(treino)
    if i == 0:
        numericas = treino[c.NB_NUMERICAS].to_numpy(float)
        X = c.desenho_nb(treino, (numericas.mean(axis=0), numericas.std(axis=0)), nb["aisps"])
        inicio = np.zeros(X.shape[1] + 1)
        inicio[0] = np.log(treino[c.ALVO].mean())
        erro = c.conferir_gradiente(X, treino[c.ALVO].to_numpy(float), inicio)
        assert erro < 1e-6, erro
        c.registrar("%s, janela: gradiente analítico conferido no primeiro ajuste, erro relativo %.1e" % (m, erro))
    ajustes.append(c.resumo_nb(nb, str(mes)))
    previsoes = {"modelo": modelo.predict(teste[nomes]).astype(float), "binomial_negativa": c.prever_nb(nb, teste),
                 "persistencia": referencias.loc[teste.index, "persistencia"].to_numpy(),
                 "media_movel_12": referencias.loc[teste.index, "media_movel_12"].to_numpy(),
                 "media_sazonal": referencias.loc[teste.index, "media_sazonal"].to_numpy()}
    y = teste[c.ALVO].to_numpy(float)
    linhas = teste[chave].assign(bairro=teste["bairro"].astype(str))
    for k in c.METODOS:
        mensais.append(c.linha_de_metricas(y, previsoes[k], ano=mes.year, mes=mes.month, metodo=k))
        partes.append(linhas.assign(forma="janela", metodo=k, alvo=y, previsao=previsoes[k]))
tabela = pd.concat(partes, ignore_index=True)
anuais = [c.linha_de_metricas(g["alvo"].to_numpy(float), g["previsao"].to_numpy(float), forma="janela", ano=a, metodo=k)
          for a in sorted(set(meses.year)) for k in c.METODOS
          for g in [tabela[(tabela["ano_alvo"] == a) & (tabela["metodo"] == k)]]]
c.gravar_csv(pd.DataFrame(mensais), saida / "metricas_mensais.csv")
c.gravar_csv(pd.DataFrame(anuais), saida / "metricas_janela.csv")
c.gravar_csv(tabela, saida / "previsoes_janela.csv")
c.gravar_json(ajustes, saida / "nb_janela.json")
c.registrar("%s, janela: fim; binomial negativa convergiu em %d de %d ajustes; gravados metricas_mensais.csv, "
            "metricas_janela.csv, previsoes_janela.csv e nb_janela.json"
            % (m, sum(a["convergiu"] for a in ajustes), len(ajustes)))
