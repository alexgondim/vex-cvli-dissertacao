"""Item 9 do bloco 31 (modo teste, 1 vez, depois do congelamento): a LSTM da semente 42, com os pesos da melhor
época (lstm_modelo.pt), avaliada nas 1.992 linhas de 2024, no corte fixo, sem janela expansiva. As métricas de 2023
recalculadas aqui têm de repetir as do ensaio.

Grava lstm_metricas.csv (2023 e 2024) e lstm_previsoes.csv (linha a linha, 2023 e 2024).
"""
import pandas as pd
import torch

import comum3 as c

m = c.modo()
assert m == "teste"
c.registrar("teste, item 9: início, LSTM da semente %d avaliada nas linhas de alvo em 2024" % c.SEMENTE_TESTE)
base = c.ler_base(m)
rede = c.Rede()
rede.load_state_dict(torch.load(c.PARTE3 / "lstm_modelo.pt"))
ensaio = pd.read_csv(c.PARTE3 / "lstm_sementes.csv").set_index("semente").loc[c.SEMENTE_TESTE]
linhas, previsoes = [], []
for ano in (2023, 2024):
    df = base[base["ano_alvo"] == ano]
    assert len(df) == 1992
    X, y = c.sequencias(df)
    q = c.prever(rede, X)
    metricas = c.comum.metricas(y, q)
    linhas.append({"modelo": "lstm", "semente": c.SEMENTE_TESTE, "melhor_epoca": int(ensaio["melhor_epoca"]),
                   "forma": "corte_fixo", "ano": ano, "linhas": 1992, **metricas})
    previsoes.append(c.identificacao(df).assign(alvo=y.astype(float), previsao=q))
    if ano == 2023:
        assert all(metricas[k] == ensaio[k] for k in ("mae", "rmse", "deviance")), "2023 não repete o ensaio"
        c.registrar("item 9, 2023 recalculado: repete o ensaio da semente %d (MAE %.6f, RMSE %.6f, deviance %.6f)"
                    % (c.SEMENTE_TESTE, *metricas.values()))
    else:
        c.registrar("item 9, LSTM no corte fixo de 2024: MAE %.6f, RMSE %.6f, deviance %.6f" % tuple(metricas.values()))
c.c2.gravar_csv(pd.DataFrame(linhas), c.PARTE3 / "lstm_metricas.csv")
c.c2.gravar_csv(pd.concat(previsoes, ignore_index=True), c.PARTE3 / "lstm_previsoes.csv")
c.registrar("teste, item 9: fim; gravados lstm_metricas.csv e lstm_previsoes.csv")
