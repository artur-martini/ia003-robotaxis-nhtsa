import pandas as pd
import pytest

from analise import dados
from analise.amostragem import SEMENTE, plano_amostral, sortear_aas


@pytest.fixture(scope="module")
def pop():
    return dados.carregar_populacao()


@pytest.fixture(scope="module")
def amostra():
    return dados.carregar_amostra()


def test_csv_da_populacao_bate_com_os_brutos(pop):
    recalculada = dados.montar_populacao()
    assert len(recalculada) == len(pop)
    assert recalculada["id_incidente"].tolist() == pop["id_incidente"].tolist()


def test_um_registro_por_acidente(pop):
    assert pop["id_incidente"].is_unique
    assert (pop["id_incidente"].str.strip() != "").all()


def test_tamanho_e_periodo(pop):
    assert len(pop) == 3070
    datas = pop["data"].dropna()
    assert len(datas) == len(pop) - 1  # um relatório da VinFast tem data desconhecida
    assert datas.min() == "2021-07"
    assert datas.max() == "2026-09"
    assert set(pop["arquivo"]) == set(dados.ARQUIVOS_BRUTOS)


def test_categorias_traduzidas(pop):
    assert set(pop["gravidade"]) <= set(dados.ORDEM_GRAVIDADE)
    assert set(pop["movimento"]) <= set(dados.MOVIMENTO.values())
    assert set(pop["colidiu_com"]) <= set(dados.COLIDIU_COM.values())
    assert set(pop["tipo_via"]) <= set(dados.TIPO_VIA.values())
    assert set(pop["empresa_grupo"]) <= set(dados.ORDEM_EMPRESA)


def test_indicadores_derivados(pop):
    assert pop["parado"].eq(pop["movimento"].isin(dados.MOVIMENTOS_PARADO)).all()
    desconhecida = pop["gravidade"] == "Desconhecida"
    assert pop.loc[desconhecida, "ferido"].isna().all()
    assert not pop.loc[~desconhecida, "ferido"].isna().any()
    assert pop.loc[pop["gravidade"] == "Sem feridos", "ferido"].eq(False).all()


def test_velocidade_plausivel(pop):
    v = pop["velocidade_mph"].dropna()
    assert v.between(0, 100).all()
    assert pop["velocidade_kmh"].dropna().round(0).equals((v * dados.MPH_PARA_KMH).round(1).round(0))


def test_plano_amostral():
    plano = plano_amostral(3070)
    assert plano.n == 342


def test_amostra_e_subconjunto_sem_repeticao(pop, amostra):
    assert len(amostra) == plano_amostral(len(pop)).n
    assert amostra["id_incidente"].is_unique
    assert set(amostra["id_incidente"]) <= set(pop["id_incidente"])


def test_amostra_reproduzivel(pop, amostra):
    de_novo = sortear_aas(pop, len(amostra), SEMENTE)
    assert de_novo["id_incidente"].tolist() == amostra["id_incidente"].tolist()


def test_amostra_tem_mesmas_linhas_da_populacao(pop, amostra):
    juntas = amostra.merge(pop, on="id_incidente", suffixes=("", "_pop"))
    assert len(juntas) == len(amostra)
    pd.testing.assert_series_equal(juntas["gravidade"], juntas["gravidade_pop"], check_names=False)


def test_sorteio_aproximadamente_uniforme(pop):
    # cada acidente deve ter probabilidade n/N de entrar; conferimos em 400 sorteios
    contagem = pd.Series(0, index=pop["id_incidente"])
    for s in range(400):
        contagem[sortear_aas(pop, 342, s)["id_incidente"]] += 1
    freq = contagem / 400
    assert freq.mean() == pytest.approx(342 / len(pop), rel=1e-9)
    assert freq.between(0.02, 0.25).all()
