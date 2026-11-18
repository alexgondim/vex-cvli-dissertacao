"""Prepara as explicações que o painel lê.

Lê `resultados/contribuicoes_corte_fixo.csv`, `resultados/escolha.json` e a base por bairro e mês. Grava em
`resultados/`:
- `explicacoes.csv`: as 11.952 linhas, com bairro, mês-alvo, divisão (treino, validação ou teste), RISP, valor de
  base, margem, estimativa em registros esperados, os 30 valores (valor_<atributo>) e as 30 contribuições
  (contrib_<atributo>), na escala do log da contagem esperada. O alvo não vai: o observado da explicação local vem da
  base;
- `explicacoes_info.json`: modelo, períodos de treino, validação e teste, configuração, ordem na grade, atributos e
  escala.

Da raiz do repositório: python modelo/exportar_para_painel.py
"""
import json

import pandas as pd

import comum

DESTINO = comum.PASTA
nomes = comum.atributos()
base = comum.ler_base().reset_index(drop=True)
c = pd.read_csv(comum.PASTA / "contribuicoes_corte_fixo.csv")
assert (c["bairro"].to_numpy() == base["bairro"].astype(str).to_numpy()).all()
assert (c[["ano_alvo", "mes_alvo"]].to_numpy() == base[["ano_alvo", "mes_alvo"]].to_numpy()).all()
saida = c[["bairro", "ano_alvo", "mes_alvo", "divisao"]].assign(risp=base["risp"].astype(str))
saida = saida.assign(valor_base=c["valor_base"], margem=c["margem"], estimativa=c["estimativa"])
for a in nomes:
    saida["valor_" + a] = base[a].astype(str) if a in comum.CATEGORICAS else base[a]
for a in nomes:
    saida["contrib_" + a] = c["contrib_" + a]
DESTINO.mkdir(exist_ok=True)
saida.to_csv(DESTINO / "explicacoes.csv", index=False)
escolha = json.loads(comum.ESCOLHA.read_text(encoding="utf-8"))
rodada = dict(modelo="corte fixo", treino="alvos de 2019 a 2022 (7.968 linhas)",
              validacao="alvos de 2023", teste="alvos de 2024", configuracao=escolha["configuracao"],
              ordem_na_grade=escolha["ordem"], atributos=nomes, escala="log da contagem esperada")
(DESTINO / "explicacoes_info.json").write_text(json.dumps(rodada, indent=1, ensure_ascii=False), encoding="utf-8")
comum.registrar("painel: explicacoes.csv (%d linhas, sha256 %s) e explicacoes_info.json (sha256 %s)"
                % (len(saida), comum.sha256(DESTINO / "explicacoes.csv"), comum.sha256(DESTINO / "explicacoes_info.json")))
