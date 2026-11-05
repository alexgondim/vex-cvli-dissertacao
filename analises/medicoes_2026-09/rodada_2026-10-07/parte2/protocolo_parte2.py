"""Grava o protocolo da Parte 2 (protocolo_parte2.md), com data e hora, antes de qualquer conta com alvo de 2024.
Recusa sobrescrever."""
import numpy
import scipy
import xgboost

import comum2 as c

quando = c.agora()
texto = f"""# Protocolo da Parte 2 da rodada de 07/10 (bloco 30)

Gravado em {quando}, antes de qualquer conta com alvo de 2024.

- **Rodada:** `analises/medicoes_2026-09/rodada_2026-10-07/`, configuração 3 da grade (taxa de aprendizado 0,05, 100 árvores, profundidade 3, peso mínimo por folha 20), congelada em `CONGELADO.tsv` de 07/10 (sha256 `{c.sha256(c.RODADA / "CONGELADO.tsv")}`).
- **Base:** base bairro-mês de 27/09 (sha256 `{c.comum.SHA_BASE}`), conferida pelo sha256 a cada leitura.
- **Separação das partes:** nada da Parte 1 muda; a Parte 2 grava só em `parte2/`.
- **Bibliotecas:** XGBoost {xgboost.__version__}, SciPy {scipy.__version__}, NumPy {numpy.__version__}; semente 42.

## a. Referências ingênuas

Nenhuma é ajustada; cada uma usa só o histórico do próprio bairro.

- **Persistência:** a contagem do mês de referência, `cvli_lag1`.
- **Média móvel de 12 meses:** `cvli_ma12`.
- **Média sazonal:** a média das contagens do bairro no mesmo mês do alvo, em todos os anos anteriores ao do alvo, desde 2018.
  - A contagem de (bairro, ano, mês) é o alvo da linha desse mês.
  - A de 2018, que não é alvo de linha nenhuma, é a `cvli_lag12` da linha do mesmo mês de 2019.
  - Para alvo em 2019, só 2018 entra; para alvo em 2024, de 2018 a 2023.

## b. Binomial negativa

- **Atributos:** `cvli_lag1` a `cvli_lag12`, `month_sin_alvo`, `month_cos_alvo`, `year_norm_alvo` e `time_index`, e a AISP em 15 indicadoras (a AISP 1 é a referência), com intercepto: 32 colunas.
  - As 16 numéricas são padronizadas pela média e pelo desvio padrão das linhas de treino de cada ajuste. Isso só melhora o condicionamento; a estimativa de máxima verossimilhança da média não muda.
  - O posto da matriz é conferido em cada ajuste e tem de ser 32.
- **Biblioteca:** SciPy {scipy.__version__} (`scipy.optimize.minimize`, método L-BFGS-B) e NumPy {numpy.__version__}. O `statsmodels` não está no ambiente e não é instalado.
  - A log-verossimilhança e o gradiente são analíticos, em `comum2.py`.
  - O gradiente é conferido contra o numérico (diferença central, passo 10⁻⁶) no primeiro ajuste de cada programa, com erro relativo máximo abaixo de 10⁻⁶.
- **Parametrização:** NB2, ligação log, média mu = exp(X·beta), variância mu + mu²/r (alpha = 1/r). Parâmetros (beta, log r), por máxima verossimilhança. A função minimizada é menos a log-verossimilhança média.
- **Valores iniciais:** beta = 0, intercepto = log da média do alvo no treino, log r = 0.
- **Iterações:** até {c.NB_MAXITER:,} por rodada do otimizador, com gtol de 10⁻¹⁰ e ftol 0.
- **Convergência:** o maior |gradiente| da log-verossimilhança média, no ponto final, fica abaixo de 10⁻⁶. A bandeira de sucesso do otimizador não conta, porque com ftol 0 ele costuma parar na busca de linha já no ótimo numérico.
- **Quando não converge:** recomeça do último ponto, até {c.NB_RECOMECOS} vezes. Se ainda não convergir:
  - a previsão sai do último ponto;
  - o ajuste fica registrado como não convergido em `binomial_negativa.json`;
  - o método fica nas tabelas e é marcado no relatório.
  Nenhuma métrica é olhada para decidir isso.
- **Ajustes:** no corte fixo, 1 ajuste, com as 7.968 linhas de alvo de 2019 a 2022; na janela, 1 ajuste por mês, com todas as linhas de alvo anterior ao mês.
- **Conferência anterior a este protocolo** (no log), só com linhas de treino (alvos até nov/2023), sem previsão e sem métrica: o corte fixo e as 24 janelas de 2022 e 2023 convergiram por este critério.

## c. Métricas

- MAE, RMSE e deviance de Poisson média.
- Piso de 0,01 só na deviance, para o modelo, a binomial negativa, as 3 referências ingênuas e as variantes da ablação e da sensibilidade.
- MAE e RMSE sem piso.

## d. As 3 formas de medir 2024

**(i) Corte fixo**
- **Modelo:** o do corte fixo de 07/10, sem reajuste, com as estimativas já gravadas em `contribuicoes_corte_fixo.csv` (coluna `estimativa`).
- **Binomial negativa:** ajustada nas 7.968 linhas de alvo de 2019 a 2022.
- **Avaliação:** tudo nas 1.992 linhas de alvo em 2024.

**(ii) Janela expansiva**
- **Modelo:** 36 reajustes, de jan/2022 a dez/2024, com a configuração 3 e os 30 atributos. Cada um é treinado com todas as linhas de alvo anterior ao mês previsto, desde jan/2019, e prevê as 166 linhas do mês.
- **Binomial negativa:** reajustada a cada mês do mesmo jeito.
- **Referências ingênuas:** nos mesmos 36 meses.
- **Métricas:** por ano (2022, 2023 e 2024) e por mês, sem média dos 36 meses.

**(iii) Ablação e sensibilidade territorial, no corte fixo**
- **Treino e avaliação:** cada variante é treinada nas 7.968 linhas de alvo de 2019 a 2022, com a configuração 3, e avaliada nas 1.992 linhas de 2024. Só as 3 métricas, sem contribuições.
- **Ablação:**
  - M1: bloco criminal, 16 atributos;
  - M2: mais o temporal, 20;
  - M3: mais o espacial (bairro, AISP e RISP), 23;
  - M4: os 30.
- **Sensibilidade territorial:**
  - MA: os 27 não espaciais mais o bairro, 28;
  - MB: os não espaciais mais a RISP, 28;
  - MC: os não espaciais mais o bairro e a RISP, 29.

## e. Ordem das etapas

1. **Protocolo:** este arquivo.
2. **Ensaio sem 2024:** as 3 formas com 2023 no lugar de 2024 no corte fixo e na ablação e sensibilidade, e só com os 24 meses de 2022 e 2023 na janela. As linhas de alvo em 2024 saem da base logo depois da leitura. Os números da conferência do despacho têm de repetir até a 6ª casa, e a binomial negativa tem de convergir no corte fixo.
3. **Congelamento:** `CONGELADO_parte2.tsv`, com o sha256 deste protocolo e de todos os programas da Parte 2, com data e hora. O teste só roda com os arquivos iguais aos congelados.
4. **Teste:** as 3 formas com 2024, 1 vez. Conferências automáticas:
   - o M4 da ablação repete até a 6ª casa as métricas do modelo no corte fixo de 2024;
   - toda linha de `metricas.csv` tem 1.992 linhas de alvo, e toda linha de `metricas_mensais.csv`, 166;
   - as métricas da janela em 2022 e 2023 repetem as do ensaio.

Programas: `comum2.py`, `protocolo_parte2.py`, `forma_corte_fixo.py`, `forma_janela.py`, `forma_ablacao.py`, `juntar_teste.py`, `congelar_parte2.py` e `exportar_metricas.py`.

Se um erro de programa for descoberto depois de vista alguma métrica de 2024, a execução para e o erro é reportado, sem correção e sem nova rodada. Depois de vista alguma métrica de 2024, nada deste protocolo muda.
"""
with open(c.PROTOCOLO, "x", encoding="utf-8") as f:
    f.write(texto)
c.registrar("protocolo: protocolo_parte2.md gravado em %s (sha256 %s), antes de qualquer conta com alvo de 2024"
            % (quando, c.sha256(c.PROTOCOLO)))
