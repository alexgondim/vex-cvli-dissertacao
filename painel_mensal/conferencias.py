"""Conferências 1 a 4 do despacho do mock, sobre as mesmas funções que a tela usa, e o sha256
de cada arquivo lido. A conferência 5 (busca de termos) e a leitura da tela ficam em capturas.py.

Da raiz do projeto: python painel_mensal/conferencias.py
"""
import hashlib

import pandas as pd

import dados

HASH_ESPERADO = {"eventos": "1cb725a9", "base_bairro_mes": "2a828800"}


def sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def main():
    print("arquivos lidos")
    for nome, caminho in dados.ARQUIVOS.items():
        h = sha256(caminho)
        esperado = HASH_ESPERADO.get(nome)
        marca = "" if esperado is None else ("  confere com o despacho" if h.startswith(esperado) else "  DIVERGE do despacho")
        print("  %s  %s%s" % (caminho.relative_to(dados.RAIZ).as_posix(), h, marca))

    eventos, base, geo = dados.ler_eventos(), dados.ler_base(), dados.ler_geojson()
    atributos = dados.atributos_do_modelo()
    bairros = dados.ordenar_bairros(base["bairro"].unique())
    risp_de = dados.risp_por_bairro(base)
    print("eventos %d; base bairro-mês %d linhas; %d bairros; %d polígonos; %d atributos"
          % (len(eventos), len(base), len(bairros), len(geo["features"]), len(atributos)))
    print("  ano/mês do eventos iguais à data: %s" % bool(
        ((eventos["periodo"].dt.year == eventos["ano"]) & (eventos["periodo"].dt.month == eventos["mes"])).all()))
    diverge = (eventos["risp"] != eventos["bairro"].map(risp_de)).sum()
    print("  registros cuja risp no eventos difere da risp do bairro na base: %d" % diverge)

    todas = (list(dados.NATUREZAS), list(dados.TURNOS), list(dados.DIAS))
    inteiro = (dados.MESES[0], dados.MESES[-1])

    f = dados.filtrar(eventos, inteiro, *todas)
    c = dados.contagem_por_bairro(f, bairros)
    por_risp = dados.tabela_mapa(c, risp_de, "risp").groupby("risp")["valor"].first()
    print("1. sem filtro, período inteiro: soma dos %d bairros = %d; registros fora dos 166 = %d"
          % (len(c), c.sum(), len(f) - c.sum()))
    print("   por RISP: %s; soma = %d" % (", ".join("%s %d" % kv for kv in por_risp.items()), por_risp.sum()))

    # mar/2024 é o caso do despacho; o caso inicial das capturas passou a ser o de dados.MES_INICIAL
    for alvo in (pd.Period("2024-03", freq="M"), dados.MES_INICIAL):
        linha = dados.linha_da_base(base, dados.BAIRRO_INICIAL, alvo)
        t = dados.mes_de_referencia(linha)
        janela = eventos[(eventos["bairro"] == dados.BAIRRO_INICIAL) & (eventos["periodo"] >= t - 11)
                         & (eventos["periodo"] <= t)]
        print("2. %s, alvo %s: n_vitimas_hist12 da linha = %d; mês de referência %s; registros de %s a %s no "
              "eventos = %d" % (dados.BAIRRO_INICIAL, dados.mes_rotulo(alvo), linha["n_vitimas_hist12"],
                                dados.mes_rotulo(t), dados.mes_rotulo(t - 11), dados.mes_rotulo(t), len(janela)))

    # mês de pico: target_cvli do bairro nos alvos de 2024, na base e recontado no eventos
    alvos = pd.read_csv(dados.ARQUIVOS["base_bairro_mes"], usecols=["bairro", "ano_alvo", "mes_alvo", "target_cvli"])
    serie = alvos[(alvos["bairro"] == dados.BAIRRO_INICIAL) & (alvos["ano_alvo"] == 2024)].set_index("mes_alvo")["target_cvli"]
    recontagem = (eventos[(eventos["bairro"] == dados.BAIRRO_INICIAL) & (eventos["ano"] == 2024)]
                  .groupby("mes").size().reindex(range(1, 13), fill_value=0))
    print("   %s em 2024, target_cvli por mês-alvo (jan a dez): %s; igual à recontagem no eventos: %s; máximo %d em %s"
          % (dados.BAIRRO_INICIAL, list(serie.astype(int)), bool((serie.to_numpy() == recontagem.to_numpy()).all()),
             serie.max(), [dados.mes_rotulo(pd.Period(year=2024, month=int(m), freq="M")) for m in serie[serie == serie.max()].index]))

    sem_bairro = dados.poligonos_sem_bairro(geo, bairros)
    sem_poligono = dados.bairros_sem_poligono(geo, bairros)
    print("3. polígonos com bairro: %d; sem bairro: %d (%s); bairros sem polígono: %d"
          % (len(geo["features"]) - len(sem_bairro), len(sem_bairro), "; ".join(sem_bairro), len(sem_poligono)))

    so_madrugada = dados.filtrar(eventos, inteiro, todas[0], ["madrugada"], todas[2])
    dist, totais = dados.distribuicao_turno(dados.filtrar(eventos, inteiro, todas[0], None, todas[2]), "Itapuã")
    barra = dist[(dist["escopo"] == "cidade") & (dist["turno"] == "madrugada")].iloc[0]
    print("4. só madrugada, período inteiro, cidade: %d registros de %d (%s); barra da madrugada na E3: "
          "%d (%s), denominador %d"
          % (len(so_madrugada), len(eventos), dados.pct_br(len(so_madrugada) / len(eventos)),
             barra["registros"], dados.pct_br(barra["fracao"]), totais["cidade"]))


if __name__ == "__main__":
    main()
