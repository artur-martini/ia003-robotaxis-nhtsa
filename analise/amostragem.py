"""Sorteio da amostra aleatória simples (AAS) sem reposição."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analise import estatistica as est

SEMENTE = 20261115
CONFIANCA = 0.95
MARGEM = 0.05
P_PLANEJAMENTO = 0.5


def plano_amostral(N: int) -> est.TamanhoAmostra:
    return est.tamanho_amostra_proporcao(N, confianca=CONFIANCA, margem=MARGEM, p=P_PLANEJAMENTO)


def sortear_aas(populacao: pd.DataFrame, n: int, semente: int = SEMENTE) -> pd.DataFrame:
    """AAS sem reposição: cada acidente tem a mesma probabilidade n/N de entrar.

    A população é ordenada pelo id do acidente antes do sorteio, para que a
    mesma semente gere sempre a mesma amostra.
    """
    base = populacao.sort_values("id_incidente", kind="stable").reset_index(drop=True)
    rng = np.random.default_rng(semente)
    posicoes = np.sort(rng.choice(len(base), size=n, replace=False))
    return base.iloc[posicoes].reset_index(drop=True)
