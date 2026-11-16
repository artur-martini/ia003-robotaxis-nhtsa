import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from analise import estatistica as est


def test_tamanho_amostra_populacao_finita():
    r = est.tamanho_amostra_proporcao(N=3059, confianca=0.95, margem=0.05, p=0.5)
    assert r.z == pytest.approx(1.959964, abs=1e-5)
    assert r.n0 == pytest.approx(384.146, abs=1e-3)
    assert r.n_exato == pytest.approx(341.38, abs=1e-2)
    assert r.n == 342


def test_tamanho_amostra_populacao_grande_se_aproxima_de_n0():
    r = est.tamanho_amostra_proporcao(N=10_000_000, confianca=0.95, margem=0.05, p=0.5)
    assert r.n == 385


def test_descritivas_bate_com_pandas():
    s = pd.Series([0, 0, 1, 2, 5, 10, np.nan])
    d = est.descritivas(s)
    v = s.dropna()
    assert d["n"] == 6
    assert d["faltantes"] == 1
    assert d["media"] == pytest.approx(v.mean())
    assert d["mediana"] == pytest.approx(v.median())
    assert d["desvio_padrao"] == pytest.approx(v.std(ddof=1))
    assert d["q1"] == pytest.approx(v.quantile(0.25))
    assert d["q3"] == pytest.approx(v.quantile(0.75))
    assert d["coef_variacao"] == pytest.approx(v.std(ddof=1) / v.mean())


def test_frequencias_soma_um():
    f = est.frequencias(pd.Series(["a", "b", "a", "c", "a"]))
    assert list(f.columns) == ["categoria", "frequencia", "proporcao"]
    assert f["frequencia"].sum() == 5
    assert f["proporcao"].sum() == pytest.approx(1.0)
    assert f.iloc[0]["categoria"] == "a"


def test_ic_proporcao_wald():
    r = est.ic_proporcao(x=60, n=150, confianca=0.95)
    p = 0.4
    ep = math.sqrt(p * (1 - p) / 150)
    assert r.estimativa == pytest.approx(p)
    assert r.erro_padrao == pytest.approx(ep)
    assert r.inferior == pytest.approx(p - 1.959964 * ep, abs=1e-6)
    assert r.superior == pytest.approx(p + 1.959964 * ep, abs=1e-6)


def test_teste_proporcao_unilateral_superior():
    # exemplo construído: p = 0,6 com n = 100 contra pi0 = 0,5
    r = est.teste_proporcao(x=60, n=100, pi0=0.5, alternativa="maior", alfa=0.05)
    assert r.estatistica == pytest.approx(2.0)
    assert r.valor_p == pytest.approx(1 - stats.norm.cdf(2.0))
    assert r.rejeita_h0 is True
    assert r.critico == pytest.approx(1.644854, abs=1e-5)


def test_teste_proporcao_unilateral_inferior_nao_rejeita():
    r = est.teste_proporcao(x=27, n=100, pi0=0.28, alternativa="menor", alfa=0.05)
    assert r.estatistica < 0
    assert r.valor_p == pytest.approx(stats.norm.cdf(r.estatistica))
    assert r.rejeita_h0 is False


def test_teste_proporcao_bilateral():
    r = est.teste_proporcao(x=40, n=100, pi0=0.5, alternativa="diferente", alfa=0.05)
    assert r.valor_p == pytest.approx(2 * stats.norm.cdf(-2.0))


def test_ic_media_t():
    rng = np.random.default_rng(1)
    x = rng.normal(10, 3, 40)
    r = est.ic_media(x, confianca=0.95)
    li, ls = stats.t.interval(0.95, df=39, loc=x.mean(), scale=stats.sem(x))
    assert r.inferior == pytest.approx(li)
    assert r.superior == pytest.approx(ls)
    assert r.graus_liberdade == 39


@pytest.mark.parametrize("alternativa,scipy_alt", [("menor", "less"), ("maior", "greater"), ("diferente", "two-sided")])
def test_teste_t_uma_amostra_bate_com_scipy(alternativa, scipy_alt):
    rng = np.random.default_rng(2)
    x = rng.exponential(6, 300)
    r = est.teste_t_uma_amostra(x, mu0=6.5, alternativa=alternativa, alfa=0.05)
    ref = stats.ttest_1samp(x, 6.5, alternative=scipy_alt)
    assert r.estatistica == pytest.approx(ref.statistic)
    assert r.valor_p == pytest.approx(ref.pvalue)
    assert r.graus_liberdade == 299


@pytest.mark.parametrize("alternativa,scipy_alt", [("maior", "greater"), ("diferente", "two-sided")])
def test_teste_welch_bate_com_scipy(alternativa, scipy_alt):
    rng = np.random.default_rng(3)
    a = rng.exponential(7, 35)
    b = rng.exponential(6, 300)
    r = est.teste_welch(a, b, alternativa=alternativa, alfa=0.05)
    ref = stats.ttest_ind(a, b, equal_var=False, alternative=scipy_alt)
    assert r.estatistica == pytest.approx(ref.statistic)
    assert r.valor_p == pytest.approx(ref.pvalue)
    assert r.graus_liberdade == pytest.approx(ref.df)
    assert r.estimativa == pytest.approx(a.mean() - b.mean())


def test_ic_diferenca_welch():
    rng = np.random.default_rng(4)
    a = rng.normal(7, 2, 30)
    b = rng.normal(6, 3, 200)
    r = est.ic_diferenca_medias_welch(a, b, confianca=0.95)
    ref = stats.ttest_ind(a, b, equal_var=False).confidence_interval(0.95)
    assert r.inferior == pytest.approx(ref.low)
    assert r.superior == pytest.approx(ref.high)


def test_fator_correcao_finita():
    assert est.fator_correcao_finita(N=3059, n=342) == pytest.approx(math.sqrt((3059 - 342) / 3058))


def test_alternativa_invalida():
    with pytest.raises(ValueError):
        est.teste_proporcao(x=1, n=10, pi0=0.5, alternativa="xpto")


def test_poder_teste_proporcao_contra_simulacao():
    rng = np.random.default_rng(5)
    n, pi0, pi_real = 342, 0.5, 0.53
    poder = est.poder_teste_proporcao(pi0, pi_real, n, alternativa="maior", alfa=0.05)
    x = rng.binomial(n, pi_real, 20000)
    z = (x / n - pi0) / math.sqrt(pi0 * (1 - pi0) / n)
    assert poder == pytest.approx((z > 1.644854).mean(), abs=0.02)


def test_poder_sob_h0_e_alfa():
    assert est.poder_teste_proporcao(0.5, 0.5, 300, "maior", 0.05) == pytest.approx(0.05)


def test_n_para_poder_proporcao():
    n = est.n_para_poder_proporcao(0.5, 0.53, "maior", alfa=0.05, poder=0.8)
    assert est.poder_teste_proporcao(0.5, 0.53, n, "maior", 0.05) >= 0.8
    assert est.poder_teste_proporcao(0.5, 0.53, n - 1, "maior", 0.05) < 0.8
