# experimentos/

Tem: os programas que avaliam o modelo: a validação com as quatro referências de comparação, a ablação por blocos de atributos e a sensibilidade aos atributos de território. Aponta para: `resultados/`.

Nesta tabela, "a base" é a base por bairro e mês, lida pelo `modelo/comum.py`. Todo programa que importa o `validacao_comum.py` lê também `resultados/escolha.json`. Os arquivos sem pasta ficam em `resultados/`.

| arquivo | o que faz | lê | grava |
|---|---|---|---|
| `validacao_comum.py` | Define as partes comuns: modo de execução, leitura da base, as quatro referências de comparação (persistência, média móvel de 12 meses, média sazonal e binomial negativa NB2), métricas e gravação. | `resultados/escolha.json`, e a base pelo `modelo/comum.py`. | Nada. |
| `validacao_corte_fixo.py` | Avalia o modelo treinado com os alvos de 2019 a 2022, sem reajuste, ao lado das quatro referências de comparação. | A base e `contribuicoes_corte_fixo.csv`. | `validacao_metricas_corte_fixo.csv`, `validacao_previsoes_corte_fixo.csv` e `validacao_nb_corte_fixo.json`. |
| `validacao_janela_expansiva.py` | Reajusta o modelo e a binomial negativa a cada mês, de jan/2022 ao fim da janela, e prevê as 166 linhas do mês. | A base. | `validacao_metricas_mensais.csv`, `validacao_metricas_janela.csv`, `validacao_previsoes_janela.csv` e `validacao_nb_janela.json`. |
| `validacao_juntar_metricas.py` | Junta as duas validações e confere a forma das tabelas. Se uma conferência falha, para sem gravar. | Os 7 arquivos gravados pelos 2 programas acima. | `validacao_metricas.csv`; de apoio, `previsoes_<modo>.csv` e `validacao_binomial_negativa.json`. |
| `ablacao_blocos.py` | Mede M1 a M4, somando um bloco de atributos por vez. | A base. | `ablacao_blocos_2023.csv` no modo `validacao` e `ablacao_blocos_2024.csv` no modo `teste`; de apoio, as previsões de cada modo e `ablacao_blocos_resumo.json`. |
| `sensibilidade_dobras_2023.py` | Nas 12 dobras de 2023, mede M1 a M4 e as variantes de território M4, MA, MB e MC, com as contribuições das variantes, o tau-b de Kendall entre as ordenações e o peso espacial. | A base e `resultados/escolha.json`. | `sensibilidade_metricas.csv`, `sensibilidade_contribuicoes.csv`, `sensibilidade_importancia.csv` e `sensibilidade_tau_b.csv`; de apoio, `ablacao.csv`, `sensibilidade_peso_espacial.csv` e `sensibilidade_previsoes_dobras_2023.csv`. |
| `sensibilidade_corte_fixo.py` | No corte fixo, mede M1 a M4, MA, MB e MC, só com as 3 métricas. | A base. | `sensibilidade_corte_fixo_2024.csv`; de apoio, `previsoes_ablacao_sensibilidade_2024.csv`. |
| `sensibilidade_comum_md.py` | Define as partes comuns da variante MD: as listas de atributos, o corte fixo, a média do valor absoluto das contribuições e o peso espacial. | Nada além do que o `validacao_comum.py` lê. | Nada. |
| `sensibilidade_md_dobras_2023.py` | Mede a variante MD nas 12 dobras de 2023, com as contribuições, e compara a ordenação dela com a do M4 e a da MB. | A base, `sensibilidade_importancia.csv` e `sensibilidade_contribuicoes.csv`. | `sensibilidade_md_dobras_2023.csv`, `sensibilidade_md_contribuicoes_2023.csv` e `sensibilidade_md_resumo_2023.json`; de apoio, `importancia_MD_2023.csv`. |
| `sensibilidade_md_corte_fixo.py` | Mede a variante MD no corte fixo, com avaliação em 2024. | A base, `contribuicoes_corte_fixo.csv` e `sensibilidade_md_resumo_2023.json`. | `sensibilidade_md_corte_fixo_2024.csv`; de apoio, `contribuicoes_MD_2024.csv`, `importancia_MD_2024.csv` e `resumo_MD.json`. |
| `sensibilidade_comum_variantes.py` | Define as partes comuns das variantes ME, MF e MG e as listas da MA e da MD. | Nada além do que o `validacao_comum.py` lê. | Nada. |
| `sensibilidade_variantes_dobras_2023.py` | Mede as variantes ME, MF e MG nas 12 dobras de 2023, com as contribuições. | A base e `sensibilidade_importancia.csv`. | `sensibilidade_variantes_dobras_2023.csv`, `sensibilidade_mf_contribuicoes_2023.csv` e `sensibilidade_variantes_resumo_2023.json`; de apoio, as contribuições da ME e da MG e `sensibilidade_variantes_importancia_2023.csv`. |
| `sensibilidade_variantes_corte_fixo.py` | Mede as variantes ME, MF e MG no corte fixo, só com as 3 métricas. | A base. | `sensibilidade_variantes_corte_fixo_2024.csv`, no modo `teste`. |
| `sensibilidade_tabela_8_combinacoes.py` | Monta a tabela das 8 combinações de atributos espaciais. Só lê arquivos já gravados: não ajusta modelo nem calcula métrica. | 12 arquivos dos experimentos de sensibilidade. | `sensibilidade_tabela_8_combinacoes.csv`, no modo `teste`. |

Modos. Os programas de validação, o `sensibilidade_corte_fixo.py`, o `sensibilidade_variantes_corte_fixo.py` e a tabela das 8 combinações recebem `ensaio` ou `teste`. No modo `ensaio`, as linhas de alvo em 2024 saem da base logo depois da leitura, e o ano avaliado é 2023. No modo `teste`, o ano avaliado é 2024. Os programas de validação e o `sensibilidade_corte_fixo.py` gravam em `resultados/ensaio/` no modo `ensaio` e em `resultados/` no modo `teste`. O `sensibilidade_variantes_corte_fixo.py` e a tabela das 8 combinações só imprimem no modo `ensaio`. O `sensibilidade_md_dobras_2023.py` e o `sensibilidade_variantes_dobras_2023.py` rodam só no modo `ensaio`. O `sensibilidade_md_corte_fixo.py` roda só no modo `teste`. O `ablacao_blocos.py` recebe `validacao` (as 12 dobras de 2023, sem as linhas de alvo em 2024) ou `teste` (corte fixo, com avaliação em 2024). O `sensibilidade_dobras_2023.py` não recebe modo.

Composição dos blocos. A ablação do `ablacao_blocos.py`, a das tabelas `ablacao_blocos_2023.csv` e `ablacao_blocos_2024.csv`, usa 4 blocos:
- criminal, com 16 atributos: as 12 defasagens, as 3 médias móveis e a tendência;
- temporal, com 8: seno e cosseno do mês do alvo, ano normalizado, índice do tempo e as proporções de madrugada, tarde, noite e fim de semana;
- espacial, com 3: `bairro`, `aisp` e `risp`;
- perfil, com 3: proporção de vítimas do sexo masculino, idade média e número de vítimas na janela.

M1 tem o bloco criminal; M2 soma o temporal; M3 soma o espacial; M4 soma o perfil e tem os 30 atributos. O `sensibilidade_dobras_2023.py` e o `sensibilidade_corte_fixo.py` também medem M1 a M4, mas com os blocos de `modelo/comum.py`: criminal 16, temporal 4, espacial 3 e perfil 7, com as 4 proporções no perfil.

Variantes de território, sempre com os 27 atributos não espaciais: M4 (`bairro`, `aisp` e `risp`), MA (`bairro`), MB (`risp`), MC (`bairro` e `risp`), MD (`aisp` e `risp`), ME (`bairro` e `aisp`), MF (`aisp`) e MG (nenhum). Na tabela das 8 combinações, a coluna `origem` dá o experimento que mediu cada combinação, pelo prefixo dos programas: `sensibilidade` (`sensibilidade_dobras_2023.py` e `sensibilidade_corte_fixo.py`), `sensibilidade_md` e `sensibilidade_variantes`.

Da raiz do repositório, na ordem do `README.md` da raiz, por exemplo: `python experimentos/validacao_corte_fixo.py teste`.
