# Protocolo da Parte 3 da rodada de 07/10 (bloco 31)

Gravado em 2026-10-13 18:23:50, antes de qualquer conta com alvo de 2024.

- **Rodada:** `analises/medicoes_2026-09/rodada_2026-10-07/`, configuração 3 da grade ({'learning_rate': 0.05, 'n_estimators': 100, 'max_depth': 3, 'min_child_weight': 20}), congelada em `CONGELADO.tsv` de 07/10 (sha256 `898a57c21229b73c2e255d66acb4669234599732d428814580c1a14f046aeee0`); Parte 2 congelada em `parte2/CONGELADO_parte2.tsv` (sha256 `f0815a873262ae417ba2387e054f1a8997423c72083bbefbcd5452ed74b716e2`).
- **Base:** base bairro-mês de 27/09 (sha256 `2a8288006c2323fa2f4995b82afa7e079f1944db804a0091c6ffd01224101443`), conferida pelo sha256 a cada leitura. No ensaio, as linhas de alvo em 2024 saem logo depois da leitura.
- **Reaproveitamento:** os 30 atributos, a configuração 3, o estimador, as métricas (piso de 0,01 só na deviance), as 12 dobras de 2023 e o `pred_contribs` vêm de `comum.py` e `comum2.py`, importados por `comum3.py`; nada é copiado para a parte3/ nem gravado fora dela.
- **Bibliotecas:** XGBoost 3.0.2, PyTorch 2.14.1+cpu (CPU; instalado no venv em 13/10 pelo pip, o que também trocou o setuptools de 65.5.0 para 78.1.0 e trouxe filelock, fsspec, mpmath e sympy), SciPy 1.17.1, NumPy 2.4.4.
- **Ordem:** protocolo, ensaio só com alvos até 2023 (itens 2, 3, 8 e 10), congelamento, e só então 2024, 1 vez por modelo (itens 4 e 9). Nenhuma escolha de hiperparâmetro, arquitetura, época ou semente olha 2024.

## Parte A. Variante MD, sem o identificador de bairro

**Item 1. Definição.** MD usa 29 atributos: os 30 menos `bairro`. `aisp` e `risp` continuam categóricas nativas. Configuração 3, semente 42, como MA, MB e MC da Parte 1.

**Item 2. Validação de 2023**, nas 12 dobras da janela expansiva de `comum.dobras_2023` (as mesmas de `busca.py` e `ablacao_sensibilidade.py`): MAE, RMSE e deviance nas 1.992 linhas; contribuições (`pred_contribs`, log da contagem esperada) das 1.992 previsões, cada linha pelo modelo da dobra dela; média do valor absoluto por atributo; ordenação dos 27 atributos não espaciais; tau-b de Kendall (`scipy.stats.kendalltau`, `variant="b"`, sobre as médias do valor absoluto dos 27, como em `ablacao_sensibilidade.py`) contra o M4 e contra a MB, com as médias deles lidas de `sensibilidade_importancia.csv` da Parte 1; e o peso espacial = (média|φ| de `aisp` + média|φ| de `risp`) / soma das médias|φ| dos 29. Para comparação, a mesma definição de peso espacial é calculada para o M4 (bairro, aisp e risp sobre 30) e a MB (risp sobre 28) a partir de `contribuicoes_sensibilidade.csv` da Parte 1.

Acréscimo declarado: a MD também é medida no corte fixo de 2023 (treino 2019 a 2022, avaliação em 2023, só as 3 métricas), para a tabela comparativa do item 12 ter a MD na mesma forma da LSTM. Não é escolha de nada.

**Item 3. Conferência de reprodução**, antes de seguir: a MB pelas 2 rotinas, nas 12 dobras (rotina da Parte 1; tem de repetir deviance 1,044035 até a 6ª casa, e as 3 métricas de `sensibilidade_metricas.csv`, com as previsões iguais bit a bit em float32 às de `previsoes_ablacao_sensibilidade.csv`) e no corte fixo avaliado em 2023 (rotina da Parte 2; tem de repetir MAE 0,587315, RMSE 0,852285 e deviance 1,048184 até a 6ª casa). Se não repetir, parar e reportar.

**Item 4. Teste, 1 vez, depois do congelamento:** MD treinada nas 7.968 linhas de alvo de 2019 a 2022 e avaliada nas 1.992 linhas de 2024, com as 3 métricas e as contribuições nessas linhas, e os mesmos resumos do item 2. O tau-b de 2024 é contra o M4 do corte fixo (`contribuicoes_corte_fixo.csv` da Parte 1, linhas de 2024); a MB não tem contribuições em 2024 (a Parte 2 não as gravou), então o tau-b MD × MB só existe em 2023.

## Parte B. Viabilidade de uma LSTM

**Item 5. Biblioteca.** PyTorch 2.14.1+cpu, CPU. Semente em `random`, `numpy` e `torch`; `torch.use_deterministic_algorithms(True)`; 1 fio de execução, para a conta repetir bit a bit.

**Item 6. Dados.** Para cada linha da base bairro-mês (mesmo alvo, mesma divisão por ano), a entrada é a sequência dos 12 meses anteriores, `cvli_lag12` até `cvli_lag1`, nessa ordem (do mais antigo ao mais recente), 1 contagem por passo, sem escala, mais o seno e o cosseno do mês de calendário de cada passo (sen(2π·mês/12) e cos(2π·mês/12), a fórmula de `month_sin_alvo` e `month_cos_alvo` da base), calculados do mês-alvo da linha e do deslocamento do passo: o passo k (0 a 11) é o mês-alvo menos 12 − k meses, e o último passo é o mês de referência. Nenhum outro atributo entra; o bairro não entra. É uma LSTM sobre o histórico de contagens, comparável ao bloco criminal mais o temporal (M2).

**Item 7. Arquitetura e treino**, fixados aqui, antes de qualquer ajuste: 1 camada LSTM com 32 unidades (`torch.nn.LSTM`, 3 entradas por passo), 1 camada linear da saída do último passo para 1 valor, contagem esperada = exp(linear); perda = log-verossimilhança negativa de Poisson (`torch.nn.PoissonNLLLoss(log_input=True)`, com a saída linear como logaritmo da taxa); Adam com taxa de aprendizado 0.001; lotes de 256 linhas, embaralhadas a cada época por gerador semeado; no máximo 200 épocas. Treino com os alvos de 2019 a 2022 (7.968 linhas). Parada antecipada pela deviance média (piso de 0,01) nas 1.992 linhas de alvo em 2023, com paciência de 20 épocas: a melhor época é a de menor deviance (a primeira, em empate), e os pesos dela são os guardados; as linhas de 2023 servem só para a parada. Inicialização padrão do PyTorch sob a semente. A grade é esta 1 configuração; não há busca.

**Item 8. Ensaio:** 3 execuções, com sementes 42, 43, 44, cada uma com a melhor época e a deviance, o MAE e o RMSE em 2023. **A semente 42 é a que vai ao teste**, anotado aqui antes de olhar os resultados das 3. A diferença entre as 3 vai ao relatório como medida de instabilidade. Os pesos da melhor época da semente 42 são gravados em `lstm_modelo.pt` no ensaio e entram no congelamento.

**Item 9. Teste, 1 vez, depois do congelamento:** a LSTM da semente 42, com os pesos de `lstm_modelo.pt`, avaliada nas 1.992 linhas de 2024 (corte fixo), sem janela expansiva. As métricas de 2023 recalculadas no teste têm de repetir as do ensaio.

**Item 10. Comparação das explicações**, só na validação de 2023 e só por importância por permutação (sem SHAP para a LSTM). 5 permutações das 1.992 linhas de 2023, sorteadas 1 vez (`numpy.random.default_rng(42)`) e reaproveitadas em todo atributo e nos 2 modelos. Para a LSTM (`lstm_modelo.pt`): embaralhar a contagem de cada um dos 12 passos, um de cada vez, mantendo o seno e o cosseno, e medir o aumento da deviance média em relação à deviance sem permutação; importância = média dos 5 aumentos. Para o M4 das 12 dobras: os modelos das dobras não foram gravados na Parte 1 e são reajustados com a configuração 3 (as previsões têm de repetir bit a bit, em float32, as de `previsoes_ablacao_sensibilidade.csv`, modelo M4, e as 3 métricas, as de `escolha.json`); embaralhar cada um dos 30 atributos, um de cada vez, nas mesmas 1.992 linhas, com o modelo da dobra de cada linha, e medir o aumento do mesmo jeito; o relatório mostra onde as 12 defasagens ficam na ordem geral dos 30. O tau-b entre as 2 ordenações das 12 defasagens (passo k da LSTM ↔ `cvli_lag{12 − k}`) é calculado sobre os aumentos médios.

## Congelamento e teste

`CONGELADO_parte3.tsv` com o sha256 deste protocolo, de todos os programas da parte3/ e de `lstm_modelo.pt`, com data e hora. O teste só roda com os arquivos iguais aos congelados. Programas: `comum3.py`, `protocolo_parte3.py`, `md_validacao.py`, `lstm_ensaio.py`, `permutacao.py`, `congelar_parte3.py`, `md_teste.py` e `lstm_teste.py`.

Se um erro de programa for descoberto depois de vista alguma métrica de 2024, a execução para e o erro é reportado, sem correção e sem nova rodada. Depois de vista alguma métrica de 2024, nada deste protocolo muda.
