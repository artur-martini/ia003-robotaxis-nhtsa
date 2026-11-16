"""Gera dados/populacao_ads.csv a partir dos CSVs brutos da NHTSA."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analise.dados import ARQ_POPULACAO, montar_populacao  # noqa: E402


def main() -> None:
    pop = montar_populacao()
    pop.to_csv(ARQ_POPULACAO, index=False)
    print(f"população: {len(pop)} acidentes distintos -> {ARQ_POPULACAO}")
    print(pop.groupby("arquivo").size().to_string())


if __name__ == "__main__":
    main()
