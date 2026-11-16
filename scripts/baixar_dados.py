"""Baixa os CSVs de incidentes com ADS da NHTSA (Standing General Order 2021-01)."""

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analise.dados import ARQUIVOS_BRUTOS, PASTA_BRUTOS  # noqa: E402


def main() -> None:
    PASTA_BRUTOS.mkdir(parents=True, exist_ok=True)
    for nome, url in ARQUIVOS_BRUTOS.values():
        destino = PASTA_BRUTOS / nome
        print(f"baixando {url}")
        pedido = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(pedido) as resp:
            destino.write_bytes(resp.read())
        print(f"  salvo em {destino}")


if __name__ == "__main__":
    main()
