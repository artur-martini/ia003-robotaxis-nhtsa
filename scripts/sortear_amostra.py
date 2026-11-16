"""Calcula o tamanho da amostra e sorteia a AAS, salvando dados/amostra_ads.csv."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analise.amostragem import SEMENTE, plano_amostral, sortear_aas  # noqa: E402
from analise.dados import ARQ_AMOSTRA, carregar_populacao  # noqa: E402


def main() -> None:
    pop = carregar_populacao()
    plano = plano_amostral(len(pop))
    print(f"N = {plano.N}; z = {plano.z:.4f}; n0 = {plano.n0:.2f}; n = {plano.n_exato:.2f} -> {plano.n}")
    amostra = sortear_aas(pop, plano.n, SEMENTE)
    amostra.to_csv(ARQ_AMOSTRA, index=False)
    print(f"amostra: {len(amostra)} acidentes (semente {SEMENTE}) -> {ARQ_AMOSTRA}")


if __name__ == "__main__":
    main()
