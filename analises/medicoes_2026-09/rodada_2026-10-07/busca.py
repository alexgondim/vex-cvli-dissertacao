"""Item 4 do bloco 26: busca dos hiperparâmetros, só em 2023.

As 36 configurações do protocolo, nas 12 dobras de 2023, com os 30 atributos. Por configuração: MAE, RMSE e
deviance de Poisson média (piso de 0,01 só na deviance) nas 1.992 previsões. Escolha pelo critério do protocolo.
Cada configuração é gravada ao terminar (busca_configuracoes.csv e previsoes_busca.csv); uma configuração já
gravada é pulada, e a busca pode ser retomada.
"""
import json
import time

import pandas as pd

import comum

assert comum.PROTOCOLO.exists(), "o protocolo vem antes do primeiro ajuste"
TABELA = comum.PASTA / "busca_configuracoes.csv"
PREVISOES = comum.PASTA / "previsoes_busca.csv"

base = comum.ler_base()
feitas = set(pd.read_csv(TABELA)["ordem"]) if TABELA.exists() else set()
comum.registrar("etapa 3, busca: início (%d de 36 já gravadas)" % len(feitas))
for i, config in enumerate(comum.GRADE, 1):
    if i in feitas:
        continue
    t0 = time.perf_counter()
    prev, _ = comum.prever_dobras(base, comum.atributos(), config)
    m = comum.metricas(prev[comum.ALVO], prev["previsao"])
    linha = dict(ordem=i, **config, padrao=config == comum.PADRAO, **m, segundos=round(time.perf_counter() - t0, 1))
    prev.insert(0, "ordem", i)
    # a previsão vai antes da tabela: uma configuração só conta como feita com as duas gravadas
    prev.rename(columns={comum.ALVO: "alvo"}).to_csv(PREVISOES, mode="a", header=not PREVISOES.exists(), index=False)
    pd.DataFrame([linha]).to_csv(TABELA, mode="a", header=not TABELA.exists(), index=False)
    comum.registrar("busca %2d/36 %s: MAE %.6f, RMSE %.6f, deviance %.6f (%.1f s)"
                    % (i, config, m["mae"], m["rmse"], m["deviance"], linha["segundos"]))

tabela = pd.read_csv(TABELA).sort_values("ordem")
assert tabela["ordem"].tolist() == list(range(1, 37))
assert len(pd.read_csv(PREVISOES)) == 36 * 1992
tabela["deviance_6_casas"] = tabela["deviance"].round(comum.CASAS_DO_EMPATE)
ranking = tabela.sort_values(["deviance_6_casas", "ordem"]).reset_index(drop=True)
escolhida = ranking.iloc[0]
posicao_padrao = int(ranking.index[ranking["padrao"]][0]) + 1
empatadas = int((tabela["deviance_6_casas"] == escolhida["deviance_6_casas"]).sum())
config = {k: (float(escolhida[k]) if k == "learning_rate" else int(escolhida[k])) for k in comum.PADRAO}
saida = dict(ordem=int(escolhida["ordem"]), configuracao=config, mae=float(escolhida["mae"]),
             rmse=float(escolhida["rmse"]), deviance=float(escolhida["deviance"]), empatadas_na_6a_casa=empatadas,
             posicao_do_padrao_no_ranking=posicao_padrao, ordem_do_padrao_na_grade=comum.GRADE.index(comum.PADRAO) + 1,
             coincide_com_padrao=config == comum.PADRAO)
comum.ESCOLHA.write_text(json.dumps(saida, indent=1, ensure_ascii=False), encoding="utf-8")
ranking.insert(0, "posicao", range(1, 37))
ranking.to_csv(comum.PASTA / "busca_ranking.csv", index=False)
comum.registrar("busca: escolhida a configuração %d %s, deviance %.6f; empatadas na 6ª casa %d; padrão na posição %d de 36"
                % (saida["ordem"], config, saida["deviance"], empatadas, posicao_padrao))
