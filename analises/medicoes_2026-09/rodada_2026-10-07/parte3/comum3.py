"""Partes comuns da Parte 3 da rodada de 07/10 (bloco 31): a variante MD (sem o bairro) e a LSTM de viabilidade.

Reaproveita o comum2.py da parte2/ (base de 27/09 conferida pelo sha256, configuração 3 da grade, estimador,
métricas com piso de 0,01 só na deviance, dobras de 2023, pred_contribs, gravação sem sobrescrever) e só
acrescenta o que a Parte 3 não tinha: o log, o modo de execução com os arquivos da parte3/, a lista de atributos
da MD, e os dados, a rede e o treino da LSTM. Nada é gravado em parte2/ nem na pasta de cima.

Da raiz do repositório, sem bytecode:
PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 ./venv/Scripts/python.exe analises/medicoes_2026-09/rodada_2026-10-07/parte3/<programa>.py ensaio|teste

No ensaio, as linhas de alvo em 2024 saem da base logo depois da leitura (comum2.ler_base); no teste, o programa só
roda com o CONGELADO_parte3.tsv gravado e com os arquivos listados nele iguais aos congelados.
"""
import datetime
import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

PARTE3 = Path(__file__).resolve().parent
RODADA = PARTE3.parent
sys.path.insert(0, str(RODADA / "parte2"))
import comum2 as c2  # noqa: E402

comum = c2.comum
LOG = PARTE3 / "rodada_parte3.log"
PROTOCOLO = PARTE3 / "protocolo_parte3.md"
CONGELADO = PARTE3 / "CONGELADO_parte3.tsv"
CONFIG = c2.CONFIG
ALVO = comum.ALVO
ANO = c2.ANO
ANOS_DE_TREINO = c2.ANOS_DE_TREINO
B = comum.BLOCOS
NAO_ESPACIAIS = comum.em_ordem(B["criminal"] + B["temporal"] + B["perfil"])
MB = comum.em_ordem(NAO_ESPACIAIS + ["risp"])
MD = comum.em_ordem(NAO_ESPACIAIS + ["aisp", "risp"])
M4 = comum.atributos()
assert len(NAO_ESPACIAIS) == 27 and len(MB) == 28 and len(MD) == 29 and len(M4) == 30
DEFASAGENS = ["cvli_lag%d" % k for k in range(1, 13)]
CHAVE = ["bairro", "ano_alvo", "mes_alvo"]

# LSTM, fixada antes de qualquer ajuste (item 7 do bloco)
PASSOS = 12
ENTRADAS = 3  # contagem do passo, seno e cosseno do mês do passo
UNIDADES = 32
TAXA = 0.001
LOTE = 256
EPOCAS = 200
PACIENCIA = 20
SEMENTES_ENSAIO = (42, 43, 44)
SEMENTE_TESTE = 42
FIOS = 1  # 1 fio de execução no torch, para a conta repetir bit a bit
PERMUTACOES = 5
SEMENTE_PERMUTACAO = 42


def agora():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def registrar(texto):
    linha = "%s  %s" % (agora(), texto)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(linha + "\n")
    print(linha, flush=True)


def sha256(caminho):
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def modo():
    """ensaio ou teste, da linha de comando, com as mesmas travas da Parte 2 sobre os arquivos da parte3/."""
    assert len(sys.argv) == 2 and sys.argv[1] in ANO, "uso: programa.py ensaio|teste"
    m = sys.argv[1]
    if m == "ensaio":
        assert PROTOCOLO.exists(), "o protocolo vem antes do ensaio"
        assert not CONGELADO.exists(), "o ensaio é anterior ao congelamento"
    else:
        assert CONGELADO.exists(), "o teste só roda depois do congelamento"
        for _, l in pd.read_csv(CONGELADO, sep="\t").iterrows():
            assert sha256(PARTE3 / l["arquivo"]) == l["sha256"], "%s mudou depois do congelamento" % l["arquivo"]
    return m


def ler_base(m):
    return c2.ler_base(m)


def corte_fixo(base, ano):
    """Treino com alvos de 2019 a 2022 (7.968 linhas) e avaliação nas 1.992 linhas do ano."""
    treino = base[base["ano_alvo"].between(*ANOS_DE_TREINO)]
    avaliadas = base[base["ano_alvo"] == ano]
    assert len(treino) == 7968 and len(avaliadas) == 1992
    return treino, avaliadas


def identificacao(df):
    return df[CHAVE].assign(bairro=df["bairro"].astype(str))


def media_abs(contribs, nomes):
    """Média do valor absoluto da contribuição de cada atributo, na escala do log da contagem esperada."""
    return contribs[nomes].abs().mean()


def peso_espacial(medias, espaciais):
    """Soma das médias do valor absoluto dos atributos espaciais do modelo dividida pela soma de todos os dele."""
    return float(medias[[a for a in espaciais if a in medias.index]].sum() / medias.sum())


# ---- LSTM ----

def semear(semente):
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(FIOS)


def sequencias(df):
    """Entrada da LSTM de cada linha: 12 passos, do mais antigo (cvli_lag12) ao mais recente (cvli_lag1), cada um com
    a contagem do passo e o seno e o cosseno do mês de calendário daquele passo. O mês do passo k (0 a 11) é o
    mês-alvo menos 12 − k meses; o último passo é o mês de referência da linha. Sem escala nas contagens."""
    alvo = pd.PeriodIndex.from_fields(year=df["ano_alvo"], month=df["mes_alvo"], freq="M")
    ref = pd.PeriodIndex.from_fields(year=df["ano_ref"], month=df["mes_ref"], freq="M")
    assert ((alvo - ref).map(lambda d: d.n) == 1).all()
    X = np.zeros((len(df), PASSOS, ENTRADAS), dtype=np.float32)
    for k in range(PASSOS):
        X[:, k, 0] = df["cvli_lag%d" % (PASSOS - k)].to_numpy(np.float32)
        mes = np.asarray([(p - (PASSOS - k)).month for p in alvo])
        X[:, k, 1] = np.sin(2 * np.pi * mes / 12)
        X[:, k, 2] = np.cos(2 * np.pi * mes / 12)
    return X, df[ALVO].to_numpy(np.float32)


class Rede(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(ENTRADAS, UNIDADES, batch_first=True)
        self.linear = nn.Linear(UNIDADES, 1)

    def forward(self, x):
        """Logaritmo da taxa de Poisson, da saída do último passo."""
        saida, _ = self.lstm(x)
        return self.linear(saida[:, -1, :]).squeeze(-1)


def prever(rede, X):
    """Contagem esperada = exp(linear)."""
    rede.eval()
    with torch.no_grad():
        return torch.exp(rede(torch.from_numpy(X))).numpy().astype(float)


def treinar(X_treino, y_treino, X_parada, y_parada, semente, registrar_epoca=None):
    """Adam com taxa TAXA, lotes de LOTE embaralhados a cada época, no máximo EPOCAS épocas, perda de Poisson com
    log_input=True sobre a saída linear. Parada antecipada pela deviance média (piso de 0,01) nas linhas de parada,
    com paciência PACIENCIA; devolve a rede com os pesos da melhor época e o histórico."""
    semear(semente)
    rede = Rede()
    perda = nn.PoissonNLLLoss(log_input=True)
    otimizador = torch.optim.Adam(rede.parameters(), lr=TAXA)
    gerador = torch.Generator().manual_seed(semente)
    Xt, yt = torch.from_numpy(X_treino), torch.from_numpy(y_treino)
    historico, melhor, melhores_pesos = [], None, None
    for epoca in range(1, EPOCAS + 1):
        rede.train()
        ordem = torch.randperm(len(Xt), generator=gerador)
        soma = 0.0
        for i in range(0, len(Xt), LOTE):
            lote = ordem[i:i + LOTE]
            otimizador.zero_grad()
            custo = perda(rede(Xt[lote]), yt[lote])
            custo.backward()
            otimizador.step()
            soma += float(custo) * len(lote)
        m = comum.metricas(y_parada, prever(rede, X_parada))
        historico.append({"semente": semente, "epoca": epoca, "perda_treino": soma / len(Xt), **m})
        if registrar_epoca:
            registrar_epoca(historico[-1])
        if melhor is None or m["deviance"] < historico[melhor - 1]["deviance"]:
            melhor = epoca
            melhores_pesos = {k: v.clone() for k, v in rede.state_dict().items()}
        elif epoca - melhor >= PACIENCIA:
            break
    rede.load_state_dict(melhores_pesos)
    return rede, pd.DataFrame(historico), melhor
