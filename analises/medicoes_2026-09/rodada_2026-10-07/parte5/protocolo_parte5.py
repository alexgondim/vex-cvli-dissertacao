"""Grava o protocolo da Parte 5 (protocolo_parte5.md), com data e hora, antes de qualquer conta. Recusa sobrescrever."""
import numpy
import pandas
import scipy
import torch
import xgboost

import comum5 as c


def lista(nomes):
    return ", ".join("`%s`" % a for a in nomes)


quando = c.agora()
R = c.RODADA
texto = f"""# Protocolo da Parte 5 da rodada de 07/10 (bloco 42)

Gravado em {quando}, antes de qualquer conta.

- **Motivo:** o orientador leu a tabela da sensibilidade territorial do capítulo e perguntou pela combinação bairro e AISP. MA (só bairro), MB (só RISP) e MC (bairro e RISP) vêm do item 11 da proposta dele, escrito quando o modelo tinha só bairro e RISP; a MD (AISP e RISP, sem o bairro) veio no bloco 31. Com os 3 atributos espaciais, faltam 3 das 8 combinações, que esta parte mede pela mesma rotina da MD.
- **Rodada:** `analises/medicoes_2026-09/rodada_2026-10-07/`, configuração 3 da grade ({c.CONFIG}), semente {c.comum.SEMENTE}, congelada em `CONGELADO.tsv` de 07/10 (sha256 `{c.sha256(R / "CONGELADO.tsv")}`). Partes congeladas: `parte2/CONGELADO_parte2.tsv` (sha256 `{c.sha256(R / "parte2" / "CONGELADO_parte2.tsv")}`), `parte3/CONGELADO_parte3.tsv` (sha256 `{c.sha256(R / "parte3" / "CONGELADO_parte3.tsv")}`) e `parte4/CONGELADO_parte4.tsv` (sha256 `{c.sha256(R / "parte4" / "CONGELADO_parte4.tsv")}`). Nada das Partes 1 a 4 muda; tudo desta parte grava em `parte5/`.
- **Base:** base bairro-mês de 27/09 (sha256 `{c.comum.SHA_BASE}`), conferida pelo sha256 a cada leitura. No ensaio, as linhas de alvo em 2024 saem logo depois da leitura.
- **Reaproveitamento:** os 30 atributos, a configuração 3, o estimador, as métricas (piso de 0,01 só na deviance), as 12 dobras de 2023 e o `pred_contribs` vêm de `comum.py` e `comum2.py`; os 27 não espaciais (`NAO_ESPACIAIS`), a MD, o corte fixo, a identificação das linhas, a média do valor absoluto e o peso espacial vêm de `comum3.py`. O `comum5.py` importa o `comum3.py` e só acrescenta o log, o modo de execução com os arquivos da `parte5/` e as listas das variantes. Nada é copiado para a `parte5/` nem gravado fora dela. Programas rodados sem bytecode.
- **Bibliotecas:** XGBoost {xgboost.__version__}, SciPy {scipy.__version__}, NumPy {numpy.__version__}, pandas {pandas.__version__}; o PyTorch ({torch.__version__}) só é importado pelo `comum3.py`, sem uso aqui.
- **Ordem:** protocolo; ensaio só com alvos até 2023 (itens 2 e 3, e o ensaio dos programas dos itens 4 e 6, avaliados em 2023); congelamento; e só então 2024, 1 vez por variante (item 4), e a tabela (item 6). Nenhuma escolha olha 2024. Não há hiperparâmetro novo: configuração 3 e semente {c.comum.SEMENTE}, as mesmas de MA a MD.

## Item 1. Definição

Sobre os 27 atributos não espaciais do `comum3.py`, na ordem do dicionário; `bairro` e `aisp` continuam categóricas nativas.

- **ME:** os 27, `bairro` e `aisp` ({len(c.VARIANTES["ME"])} atributos; espaciais: {lista(c.espaciais_de(c.VARIANTES["ME"]))}).
- **MF:** os 27 e `aisp` ({len(c.VARIANTES["MF"])} atributos; espaciais: {lista(c.espaciais_de(c.VARIANTES["MF"]))}).
- **MG:** só os 27 ({len(c.VARIANTES["MG"])} atributos; nenhum espacial).

Os 27: {lista(c.NAO_ESPACIAIS)}.

## Item 2. Conferência de reprodução

Antes de qualquer variante nova, pela rotina das 12 dobras de 2023 (`comum.prever_dobras`): a MA (os 27 e `bairro`, {len(c.MA)} atributos) tem de repetir MAE 0,576837, RMSE 0,839612 e deviance 1,008424 até a 6ª casa, e a MD (os 27, `aisp` e `risp`, {len(c.MD)} atributos), MAE 0,586980, RMSE 0,850336 e deviance 1,040701. Se não repetir, a execução para e reporta.

Registrados também, sem ser critério de parada: as previsões da MA iguais bit a bit (float32) às de `previsoes_ablacao_sensibilidade.csv` e as 3 métricas iguais às de `sensibilidade_metricas.csv` (Parte 1); as 3 métricas da MD iguais às de `parte3/sensibilidade_MD_2023.csv` e as contribuições iguais bit a bit (float32) às de `parte3/contribuicoes_MD_2023.csv`. Grava `conferencia_2023.json`.

## Item 3. Validação de 2023

Nas 12 dobras de `comum.dobras_2023` (as mesmas da Parte 1 e da MD), para ME, MF e MG:

- MAE, RMSE e deviance nas 1.992 linhas;
- contribuições (`pred_contribs`, log da contagem esperada) das 1.992 previsões, cada linha pelo modelo da dobra dela, no mesmo formato de `parte3/contribuicoes_MD_2023.csv`; registrada no log a aditividade (maior diferença entre a soma das contribuições com o valor base e o log da previsão);
- média do valor absoluto por atributo (`comum3.media_abs`) e o atributo de maior média;
- tau-b de Kendall (`scipy.stats.kendalltau`, `variant="b"`) entre as médias dos 27 não espaciais da variante e as do M4 da Parte 1 (`sensibilidade_importancia.csv`), como foi feito para a MD;
- peso espacial (`comum3.peso_espacial`): soma das médias do valor absoluto dos atributos espaciais da variante dividida pela soma das médias de todos os atributos dela (na MG, 0).

## Item 4. Teste, 1 vez, depois do congelamento

ME, MF e MG treinadas nas 7.968 linhas de alvo de 2019 a 2022 e avaliadas nas 1.992 linhas de 2024 (corte fixo, `comum3.corte_fixo`), só as 3 métricas, como MA, MB e MC na Parte 2 (`parte2/forma_ablacao.py`). No mesmo programa, antes das variantes novas, a MD pela mesma rotina tem de repetir MAE 0,570633, RMSE 0,833318 e deviance 1,022385 até a 6ª casa (números já vistos em 13/10, `parte3/sensibilidade_MD_2024.csv`); se não repetir, o programa para antes das variantes novas. Grava `conferencia_2024.json` e `sensibilidade_parte5_2024.csv`.

**Ensaio do programa do teste**, antes do congelamento: o mesmo `teste_2024.py`, no modo ensaio, roda a mesma rotina avaliada nas 1.992 linhas de 2023, sem nenhuma linha de 2024. A MD tem de repetir o corte fixo de 2023 da Parte 3 (MAE 0,590529, RMSE 0,852938 e deviance 1,043370, `parte3/sensibilidade_MD_2023.csv`). As métricas de 2023 de ME, MF e MG por essa rotina vão só ao log, servem só para conferir o programa e não entram em escolha nenhuma.

## Item 6. Tabela das 8 combinações

`tabela_combinacoes.py` só lê arquivos gravados; não ajusta modelo nem calcula métrica. Para M4, MA, MB, MC e MD da rodada e ME, MF e MG desta parte: atributos espaciais, deviance nas 12 dobras de 2023, deviance no corte fixo de 2024, peso espacial em 2023, atributo de maior média do valor absoluto em 2023 e tau-b contra o M4.

- Deviance de 2023: `sensibilidade_metricas.csv` (M4, MA, MB e MC), `parte3/sensibilidade_MD_2023.csv` (linha das dobras) e `sensibilidade_parte5_2023.csv`.
- Deviance de 2024: `parte2/ablacao_sensibilidade_2024.csv` (M4, MA, MB e MC, a mesma rotina do corte fixo das variantes), `parte3/sensibilidade_MD_2024.csv` e `sensibilidade_parte5_2024.csv`.
- Peso espacial e maior média de 2023: M4, MA, MB e MC calculados das contribuições da Parte 1 (`contribuicoes_sensibilidade.csv`), com `comum3.media_abs` e `comum3.peso_espacial`, a conta que o `parte3/md_validacao.py` fez para o M4 e a MB (os 2 têm de sair iguais aos de `parte3/resumo_MD_2023.json`; registrado no log, sem ser critério de parada); MD de `parte3/resumo_MD_2023.json`; ME, MF e MG de `resumo_parte5.json`.
- Tau-b contra o M4: `sensibilidade_tau_b.csv` (MA, MB e MC), `parte3/resumo_MD_2023.json` (MD) e `resumo_parte5.json`; no M4, 1 por definição.

No ensaio, a tabela é montada sem nenhuma coluna de 2024 e só vai ao log; no teste, com as deviances de 2024, é gravada em `tabela_8_combinacoes.csv`.

## Arquivos

Na `parte5/`: este protocolo, `CONGELADO_parte5.tsv`, `rodada_parte5.log`, os programas (`comum5.py`, `protocolo_parte5.py`, `validacao_2023.py`, `teste_2024.py`, `tabela_combinacoes.py` e `congelar_parte5.py`), `sensibilidade_parte5_2023.csv` e `sensibilidade_parte5_2024.csv` (uma linha por variante, com as 3 métricas), `contribuicoes_ME_2023.csv`, `contribuicoes_MF_2023.csv` e `contribuicoes_MG_2023.csv`, `importancia_parte5_2023.csv` (média do valor absoluto por atributo e variante) e `resumo_parte5.json` (tau-b, peso espacial e maior média de cada variante). Acrescentados, declarados aqui: `conferencia_2023.json` e `conferencia_2024.json` (conferências dos itens 2 e 4) e `tabela_8_combinacoes.csv` (item 6).

## Congelamento e teste

`CONGELADO_parte5.tsv` com o sha256 deste protocolo, de todos os programas da `parte5/` e das saídas do ensaio que o teste e a tabela leem, com data e hora. Só é gravado com a conferência do item 2 ok e com os ensaios do `teste_2024.py` e do `tabela_combinacoes.py` até o fim no log. O teste só roda com esses arquivos iguais aos congelados.

Se um erro de programa for descoberto depois de vista alguma métrica de 2024, a execução para e o erro é reportado, sem correção e sem nova rodada. Depois de vista alguma métrica de 2024, nada deste protocolo muda.
"""
with open(c.PROTOCOLO, "x", encoding="utf-8") as f:
    f.write(texto)
c.registrar("protocolo: protocolo_parte5.md gravado em %s (sha256 %s), antes de qualquer conta"
            % (quando, c.sha256(c.PROTOCOLO)))
