"""Item 8 do bloco 31 (modo ensaio): a LSTM do item 7 treinada 3 vezes, com as sementes 42, 43 e 44, nos alvos de
2019 a 2022, com parada antecipada pela deviance nas 1.992 linhas de 2023. Cada execução reporta a melhor época e
as 3 métricas em 2023. A semente 42 é a do teste, fixada no protocolo.

Grava lstm_sementes.csv, lstm_epocas.csv (o histórico das 3), lstm_previsoes_ensaio_2023.csv e lstm_modelo.pt (os
pesos da melhor época da semente 42), e confere que o modelo recarregado repete a previsão.
"""
import pandas as pd
import torch

import comum3 as c

m = c.modo()
assert m == "ensaio"
c.registrar("ensaio, item 8: início; base lida sem as linhas de 2024; PyTorch %s, %d fio" % (torch.__version__, c.FIOS))
base = c.ler_base(m)
treino, parada = c.corte_fixo(base, 2023)
Xt, yt = c.sequencias(treino)
Xp, yp = c.sequencias(parada)
assert Xt.shape == (7968, c.PASSOS, c.ENTRADAS) and Xp.shape == (1992, c.PASSOS, c.ENTRADAS)
linhas, historicos, previsoes = [], [], []
for semente in c.SEMENTES_ENSAIO:
    rede, historico, melhor = c.treinar(Xt, yt, Xp, yp, semente)
    q = c.prever(rede, Xp)
    metricas = c.comum.metricas(yp, q)
    na_epoca = historico.loc[historico["epoca"] == melhor].iloc[0]
    assert all(metricas[k] == na_epoca[k] for k in ("mae", "rmse", "deviance")), "os pesos guardados não são os da melhor época"
    linhas.append({"semente": semente, "melhor_epoca": melhor, "epocas_rodadas": len(historico),
                   "perda_treino_na_melhor": float(na_epoca["perda_treino"]), **metricas})
    historicos.append(historico)
    previsoes.append(c.identificacao(parada).assign(semente=semente, alvo=yp.astype(float), previsao=q))
    c.registrar("item 8, semente %d: melhor época %d de %d rodadas; em 2023, MAE %.6f, RMSE %.6f, deviance %.6f"
                % (semente, melhor, len(historico), metricas["mae"], metricas["rmse"], metricas["deviance"]))
    if semente == c.SEMENTE_TESTE:
        caminho = c.PARTE3 / "lstm_modelo.pt"
        assert not caminho.exists()
        torch.save(rede.state_dict(), caminho)
        outra = c.Rede()
        outra.load_state_dict(torch.load(caminho))
        assert (c.prever(outra, Xp) == q).all(), "o modelo recarregado não repete a previsão"
        c.registrar("item 8: lstm_modelo.pt gravado com os pesos da melhor época da semente %d; recarregado, repete "
                    "a previsão de 2023 bit a bit" % semente)
c.c2.gravar_csv(pd.DataFrame(linhas), c.PARTE3 / "lstm_sementes.csv")
c.c2.gravar_csv(pd.concat(historicos, ignore_index=True), c.PARTE3 / "lstm_epocas.csv")
c.c2.gravar_csv(pd.concat(previsoes, ignore_index=True), c.PARTE3 / "lstm_previsoes_ensaio_2023.csv")
dev = [l["deviance"] for l in linhas]
c.registrar("ensaio, item 8: fim; deviance em 2023 entre as 3 sementes de %.6f a %.6f (amplitude %.6f); gravados "
            "lstm_sementes.csv, lstm_epocas.csv, lstm_previsoes_ensaio_2023.csv e lstm_modelo.pt"
            % (min(dev), max(dev), max(dev) - min(dev)))
