"""Funções de estatística descritiva e inferencial usadas no trabalho.

As fórmulas seguem as da disciplina: teste Z para uma proporção (erro padrão
calculado com pi0), intervalo de Wald para a proporção, teste t para uma média
(sigma desconhecido) e teste t de Welch para duas médias independentes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

ALTERNATIVAS = ("menor", "maior", "diferente")
SIMBOLO_ALTERNATIVA = {"menor": "<", "maior": ">", "diferente": "≠"}


def _checar_alternativa(alternativa: str) -> None:
    if alternativa not in ALTERNATIVAS:
        raise ValueError(f"alternativa deve ser uma de {ALTERNATIVAS}, recebeu {alternativa!r}")


def _valor_p(estatistica: float, alternativa: str, cdf, sf) -> float:
    if alternativa == "menor":
        return float(cdf(estatistica))
    if alternativa == "maior":
        return float(sf(estatistica))
    return float(2 * sf(abs(estatistica)))


def _critico(alfa: float, alternativa: str, ppf) -> float:
    """Valor crítico (positivo) da distribuição de referência."""
    if alternativa == "diferente":
        return float(ppf(1 - alfa / 2))
    return float(ppf(1 - alfa))


# ---------------------------------------------------------------- tamanho amostral


@dataclass(frozen=True)
class TamanhoAmostra:
    N: int
    confianca: float
    margem: float
    p: float
    z: float
    n0: float
    n_exato: float
    n: int


def tamanho_amostra_proporcao(N: int, confianca: float = 0.95, margem: float = 0.05, p: float = 0.5) -> TamanhoAmostra:
    """Tamanho de amostra para estimar uma proporção numa população finita.

    n0 = z² p (1 - p) / e²   (população infinita)
    n  = n0 / (1 + (n0 - 1) / N)   (correção para população finita)

    É a mesma conta de n = N z² p q / (e² (N - 1) + z² p q), usada pela
    calculadora da USP/Bauru.
    """
    z = float(stats.norm.ppf(1 - (1 - confianca) / 2))
    n0 = z**2 * p * (1 - p) / margem**2
    n_exato = n0 / (1 + (n0 - 1) / N)
    return TamanhoAmostra(N, confianca, margem, p, z, n0, n_exato, math.ceil(n_exato))


def fator_correcao_finita(N: int, n: int) -> float:
    return math.sqrt((N - n) / (N - 1))


# ---------------------------------------------------------------- descritiva


def descritivas(serie: pd.Series) -> dict:
    v = pd.to_numeric(serie, errors="coerce")
    faltantes = int(v.isna().sum())
    v = v.dropna()
    media = float(v.mean())
    dp = float(v.std(ddof=1))
    q1, q3 = float(v.quantile(0.25)), float(v.quantile(0.75))
    return {
        "n": int(v.size),
        "faltantes": faltantes,
        "media": media,
        "mediana": float(v.median()),
        "moda": float(v.mode().iloc[0]) if v.size else float("nan"),
        "desvio_padrao": dp,
        "variancia": float(v.var(ddof=1)),
        "coef_variacao": dp / media if media else float("nan"),
        "minimo": float(v.min()),
        "q1": q1,
        "q3": q3,
        "maximo": float(v.max()),
        "amplitude_interquartil": q3 - q1,
        "assimetria": float(stats.skew(v, bias=False)),
    }


def frequencias(serie: pd.Series, ordem: list | None = None) -> pd.DataFrame:
    contagem = serie.value_counts(dropna=False)
    if ordem is not None:
        contagem = contagem.reindex([c for c in ordem if c in contagem.index])
    f = contagem.rename_axis("categoria").reset_index(name="frequencia")
    f["proporcao"] = f["frequencia"] / f["frequencia"].sum()
    return f


# ---------------------------------------------------------------- inferência


@dataclass(frozen=True)
class Intervalo:
    estimativa: float
    erro_padrao: float
    critico: float
    inferior: float
    superior: float
    confianca: float
    n: int
    graus_liberdade: float | None = None

    @property
    def margem(self) -> float:
        return self.critico * self.erro_padrao

    def contem(self, valor: float) -> bool:
        return self.inferior <= valor <= self.superior


@dataclass(frozen=True)
class ResultadoTeste:
    estimativa: float
    valor_h0: float
    erro_padrao: float
    estatistica: float
    distribuicao: str
    graus_liberdade: float | None
    valor_p: float
    alfa: float
    alternativa: str
    critico: float
    n: int

    @property
    def rejeita_h0(self) -> bool:
        return self.valor_p < self.alfa

    @property
    def regiao_critica(self) -> str:
        nome = "Z" if self.distribuicao == "normal" else "T"
        if self.alternativa == "menor":
            return f"{nome} < {-self.critico:.3f}"
        if self.alternativa == "maior":
            return f"{nome} > {self.critico:.3f}"
        return f"|{nome}| > {self.critico:.3f}"


def ic_proporcao(x: int, n: int, confianca: float = 0.95) -> Intervalo:
    """Intervalo de Wald: p ± z √(p(1-p)/n)."""
    p = x / n
    ep = math.sqrt(p * (1 - p) / n)
    z = float(stats.norm.ppf(1 - (1 - confianca) / 2))
    return Intervalo(p, ep, z, p - z * ep, p + z * ep, confianca, n)


def teste_proporcao(x: int, n: int, pi0: float, alternativa: str = "diferente", alfa: float = 0.05) -> ResultadoTeste:
    """Teste Z para uma proporção: Z = (p - pi0) / √(pi0 (1 - pi0) / n)."""
    _checar_alternativa(alternativa)
    p = x / n
    ep0 = math.sqrt(pi0 * (1 - pi0) / n)
    z = (p - pi0) / ep0
    return ResultadoTeste(
        estimativa=p,
        valor_h0=pi0,
        erro_padrao=ep0,
        estatistica=z,
        distribuicao="normal",
        graus_liberdade=None,
        valor_p=_valor_p(z, alternativa, stats.norm.cdf, stats.norm.sf),
        alfa=alfa,
        alternativa=alternativa,
        critico=_critico(alfa, alternativa, stats.norm.ppf),
        n=n,
    )


def ic_media(amostra, confianca: float = 0.95) -> Intervalo:
    """Intervalo t: x̄ ± t(n-1) s/√n."""
    x = np.asarray(pd.Series(amostra).dropna(), dtype=float)
    n = x.size
    ep = x.std(ddof=1) / math.sqrt(n)
    t = float(stats.t.ppf(1 - (1 - confianca) / 2, df=n - 1))
    m = float(x.mean())
    return Intervalo(m, float(ep), t, float(m - t * ep), float(m + t * ep), confianca, n, n - 1)


def teste_t_uma_amostra(amostra, mu0: float, alternativa: str = "diferente", alfa: float = 0.05) -> ResultadoTeste:
    """Teste t para uma média: T = (x̄ - mu0) / (s / √n), com n - 1 graus de liberdade."""
    _checar_alternativa(alternativa)
    x = np.asarray(pd.Series(amostra).dropna(), dtype=float)
    n = x.size
    gl = n - 1
    ep = x.std(ddof=1) / math.sqrt(n)
    t = (x.mean() - mu0) / ep
    return ResultadoTeste(
        estimativa=float(x.mean()),
        valor_h0=mu0,
        erro_padrao=float(ep),
        estatistica=float(t),
        distribuicao="t",
        graus_liberdade=gl,
        valor_p=_valor_p(t, alternativa, lambda v: stats.t.cdf(v, gl), lambda v: stats.t.sf(v, gl)),
        alfa=alfa,
        alternativa=alternativa,
        critico=_critico(alfa, alternativa, lambda q: stats.t.ppf(q, gl)),
        n=n,
    )


def _welch_partes(a, b):
    a = np.asarray(pd.Series(a).dropna(), dtype=float)
    b = np.asarray(pd.Series(b).dropna(), dtype=float)
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    ep = math.sqrt(va + vb)
    gl = (va + vb) ** 2 / (va**2 / (a.size - 1) + vb**2 / (b.size - 1))
    return a, b, ep, gl


def teste_welch(a, b, alternativa: str = "diferente", alfa: float = 0.05) -> ResultadoTeste:
    """Teste t de Welch: T = (x̄1 - x̄2) / √(s1²/n1 + s2²/n2), gl de Welch–Satterthwaite."""
    _checar_alternativa(alternativa)
    a, b, ep, gl = _welch_partes(a, b)
    dif = float(a.mean() - b.mean())
    t = dif / ep
    return ResultadoTeste(
        estimativa=dif,
        valor_h0=0.0,
        erro_padrao=ep,
        estatistica=t,
        distribuicao="t",
        graus_liberdade=gl,
        valor_p=_valor_p(t, alternativa, lambda v: stats.t.cdf(v, gl), lambda v: stats.t.sf(v, gl)),
        alfa=alfa,
        alternativa=alternativa,
        critico=_critico(alfa, alternativa, lambda q: stats.t.ppf(q, gl)),
        n=int(a.size + b.size),
    )


def ic_diferenca_medias_welch(a, b, confianca: float = 0.95) -> Intervalo:
    a, b, ep, gl = _welch_partes(a, b)
    dif = float(a.mean() - b.mean())
    t = float(stats.t.ppf(1 - (1 - confianca) / 2, df=gl))
    return Intervalo(dif, ep, t, dif - t * ep, dif + t * ep, confianca, int(a.size + b.size), gl)


# ---------------------------------------------------------------- poder


def poder_teste_proporcao(pi0: float, pi_real: float, n: int, alternativa: str = "maior", alfa: float = 0.05) -> float:
    """Probabilidade de rejeitar H₀ no teste Z de uma proporção quando a proporção verdadeira é pi_real."""
    _checar_alternativa(alternativa)
    ep0 = math.sqrt(pi0 * (1 - pi0) / n)
    ep1 = math.sqrt(pi_real * (1 - pi_real) / n)
    if alternativa == "maior":
        limite = pi0 + stats.norm.ppf(1 - alfa) * ep0
        return float(stats.norm.sf((limite - pi_real) / ep1))
    if alternativa == "menor":
        limite = pi0 - stats.norm.ppf(1 - alfa) * ep0
        return float(stats.norm.cdf((limite - pi_real) / ep1))
    z = stats.norm.ppf(1 - alfa / 2)
    sup, inf = pi0 + z * ep0, pi0 - z * ep0
    return float(stats.norm.sf((sup - pi_real) / ep1) + stats.norm.cdf((inf - pi_real) / ep1))


def n_para_poder_proporcao(pi0: float, pi_real: float, alternativa: str = "maior", alfa: float = 0.05, poder: float = 0.8) -> int:
    """Menor n que atinge o poder desejado (fórmula aproximada, conferida pela função de poder)."""
    za = stats.norm.ppf(1 - (alfa / 2 if alternativa == "diferente" else alfa))
    zb = stats.norm.ppf(poder)
    n = math.ceil(((za * math.sqrt(pi0 * (1 - pi0)) + zb * math.sqrt(pi_real * (1 - pi_real))) / (pi_real - pi0)) ** 2)
    while poder_teste_proporcao(pi0, pi_real, n, alternativa, alfa) < poder:
        n += 1
    while n > 1 and poder_teste_proporcao(pi0, pi_real, n - 1, alternativa, alfa) >= poder:
        n -= 1
    return n


def poder_teste_t_aproximado(diferenca: float, erro_padrao: float, alternativa: str = "maior", alfa: float = 0.05) -> float:
    """Poder aproximado (normal) de um teste de médias para uma diferença verdadeira e um erro padrão dados."""
    _checar_alternativa(alternativa)
    if alternativa == "maior":
        return float(stats.norm.sf(stats.norm.ppf(1 - alfa) - diferenca / erro_padrao))
    if alternativa == "menor":
        return float(stats.norm.cdf(-stats.norm.ppf(1 - alfa) - diferenca / erro_padrao))
    z = stats.norm.ppf(1 - alfa / 2)
    return float(stats.norm.sf(z - diferenca / erro_padrao) + stats.norm.cdf(-z - diferenca / erro_padrao))
