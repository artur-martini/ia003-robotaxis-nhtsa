"""Leitura, limpeza e harmonização dos relatórios de acidentes com ADS (NHTSA SGO 2021-01).

A NHTSA publica dois arquivos de incidentes com ADS (direção automatizada):
- histórico: relatórios enviados até junho de 2025 (acidentes de jul/2021 a jun/2025);
- atual: relatórios no formato da emenda de 2025 (acidentes de abr/2025 em diante).

Cada acidente pode ter várias versões do mesmo relatório (Report Version) e
relatórios de mais de uma empresa (mesmo Same Incident ID). A população do
trabalho é o conjunto de acidentes distintos: uma linha por Same Incident ID.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
PASTA_DADOS = RAIZ / "dados"
PASTA_BRUTOS = PASTA_DADOS / "brutos"
ARQ_POPULACAO = PASTA_DADOS / "populacao_ads.csv"
ARQ_AMOSTRA = PASTA_DADOS / "amostra_ads.csv"

URL_BASE = "https://static.nhtsa.gov/odi/ffdd/sgo-2021-01"
URL_PAGINA = "https://www.nhtsa.gov/laws-regulations/standing-general-order-crash-reporting"
ARQUIVOS_BRUTOS = {
    "Histórico (2021–2025)": ("Archive_SGO-2021-01_Incident_Reports_ADS.csv", f"{URL_BASE}/Archive-2021-2025/SGO-2021-01_Incident_Reports_ADS.csv"),
    "Atual (2025–2026)": ("SGO-2021-01_Incident_Reports_ADS.csv", f"{URL_BASE}/SGO-2021-01_Incident_Reports_ADS.csv"),
}

MPH_PARA_KMH = 1.609344

# ---------------------------------------------------------------- dicionários de tradução

GRAVIDADE = {
    "No Injuries Reported": "Sem feridos",
    "No Injured Reported": "Sem feridos",
    "Property Damage. No Injured Reported": "Sem feridos",
    "Minor": "Leve",
    "Minor W/O Hospitalization": "Leve",
    "Minor W/ Hospitalization": "Leve",
    "Moderate": "Moderada",
    "Moderate W/O Hospitalization": "Moderada",
    "Moderate W/ Hospitalization": "Moderada",
    "Serious": "Grave",
    "Serious W/O Hospitalization": "Grave",
    "Serious W/ Hospitalization": "Grave",
    "Fatality": "Fatal",
    "Unknown": "Desconhecida",
}
ORDEM_GRAVIDADE = ["Sem feridos", "Leve", "Moderada", "Grave", "Fatal", "Desconhecida"]

MOVIMENTO = {
    "Stopped": "Parado no trânsito",
    "Parked": "Estacionado",
    "Proceeding Straight": "Seguindo em frente",
    "Making Left Turn": "Virando à esquerda",
    "Making Right Turn": "Virando à direita",
    "Making U-Turn": "Fazendo retorno",
    "Changing Lanes": "Mudando de faixa",
    "Backing": "Dando ré",
    "Lane / Road Departure": "Saindo da faixa ou da via",
    "Passing": "Ultrapassando",
    "Entering Traffic": "Entrando no tráfego",
    "Merging": "Entrando no tráfego",
    "Parking Maneuver": "Manobrando para estacionar",
    "Crossing into Opposing Lane": "Invadindo a contramão",
    "Other, see Narrative": "Outro ou desconhecido",
    "Unknown": "Outro ou desconhecido",
}
MOVIMENTOS_PARADO = {"Parado no trânsito", "Estacionado"}

COLIDIU_COM = {
    "Passenger Car": "Carro de passeio",
    "SUV": "SUV",
    "Pickup Truck": "Picape",
    "Van": "Van",
    "Heavy Truck": "Caminhão",
    "Bus": "Ônibus",
    "Motorcycle": "Motocicleta",
    "First Responder Vehicle": "Veículo de emergência",
    "Non-Motorist: Cyclist": "Ciclista",
    "Non-Motorist: Pedestrian": "Pedestre",
    "Non-Motorist: Scooter - Skateboard": "Patinete ou skate",
    "Non-Motorist: Other": "Outro não motorizado",
    "Other Fixed Object": "Objeto fixo",
    "Pole / Tree": "Objeto fixo",
    "Animal": "Animal",
    "Other, see Narrative": "Outro ou desconhecido",
    "Unknown": "Outro ou desconhecido",
}

TIPO_VIA = {
    "Street": "Rua",
    "Intersection": "Cruzamento",
    "Parking Lot": "Estacionamento",
    "Parking Garage": "Estacionamento",
    "Highway / Freeway": "Rodovia",
    "Traffic Circle": "Rotatória",
    "Rural Road": "Estrada rural",
    "Unpaved Road": "Estrada rural",
    "Unknown": "Desconhecido",
}

EMPRESA_GRUPO = {
    "Waymo LLC": "Waymo",
    "Cruise LLC": "Cruise (GM)",
    "General Motors, LLC": "Cruise (GM)",
    "Zoox, Inc.": "Zoox",
    "Transdev Alternative Services": "Transdev",
    "Avride Inc.": "Avride",
    "Tesla, Inc.": "Tesla",
}
ORDEM_EMPRESA = ["Waymo", "Cruise (GM)", "Zoox", "Transdev", "Avride", "Tesla", "Outras"]

COLUNAS_FINAIS = [
    "id_incidente",
    "id_relatorio",
    "arquivo",
    "empresa",
    "empresa_grupo",
    "data",
    "ano",
    "hora",
    "estado",
    "tipo_via",
    "colidiu_com",
    "movimento",
    "parado",
    "velocidade_mph",
    "velocidade_kmh",
    "gravidade",
    "ferido",
]


def _traduzir(serie: pd.Series, mapa: dict, padrao: str) -> pd.Series:
    traduzida = serie.str.strip().map(mapa)
    nao_mapeados = serie.dropna()[traduzida.loc[serie.dropna().index].isna()].unique()
    if len(nao_mapeados):
        raise ValueError(f"categorias sem tradução: {sorted(nao_mapeados)}")
    return traduzida.fillna(padrao)


def ler_bruto(caminho: Path) -> pd.DataFrame:
    return pd.read_csv(caminho, low_memory=False, encoding="utf-8", dtype={"Same Incident ID": str, "Report ID": str})


def deduplicar(bruto: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por acidente: a última versão de cada relatório e, depois, um relatório por Same Incident ID."""
    d = bruto.sort_values(["Report ID", "Report Version"], kind="stable")
    d = d.drop_duplicates("Report ID", keep="last").copy()
    # alguns relatórios (15 da Mobileye no arquivo histórico) vêm sem Same Incident ID:
    # cada um é um acidente próprio, identificado pelo número do relatório
    sem_id = d["Same Incident ID"].fillna("").str.strip() == ""
    d.loc[sem_id, "Same Incident ID"] = "relatorio-" + d.loc[sem_id, "Report ID"]
    d = d.sort_values(["Same Incident ID", "Report Version", "Report ID"], kind="stable")
    return d.drop_duplicates("Same Incident ID", keep="last")


def harmonizar(d: pd.DataFrame, arquivo: str) -> pd.DataFrame:
    out = pd.DataFrame(index=d.index)
    out["id_incidente"] = d["Same Incident ID"]
    out["id_relatorio"] = d["Report ID"]
    out["arquivo"] = arquivo
    out["empresa"] = d["Reporting Entity"].str.strip()
    out["empresa_grupo"] = out["empresa"].map(EMPRESA_GRUPO).fillna("Outras")
    data = pd.to_datetime(d["Incident Date"], format="%b-%Y", errors="coerce")
    out["data"] = data.dt.strftime("%Y-%m")
    out["ano"] = data.dt.year.astype("Int64")
    hora = pd.to_datetime(d["Incident Time (24:00)"], format="%H:%M", errors="coerce")
    out["hora"] = hora.dt.hour.astype("Int64")
    out["estado"] = d["State"].str.strip()
    out["tipo_via"] = _traduzir(d["Roadway Type"], TIPO_VIA, "Desconhecido")
    out["colidiu_com"] = _traduzir(d["Crash With"], COLIDIU_COM, "Outro ou desconhecido")
    out["movimento"] = _traduzir(d["SV Pre-Crash Movement"], MOVIMENTO, "Outro ou desconhecido")
    out["parado"] = out["movimento"].isin(MOVIMENTOS_PARADO)
    out["velocidade_mph"] = pd.to_numeric(d["SV Precrash Speed (MPH)"], errors="coerce")
    out["velocidade_kmh"] = (out["velocidade_mph"] * MPH_PARA_KMH).round(1)
    out["gravidade"] = _traduzir(d["Highest Injury Severity Alleged"], GRAVIDADE, "Desconhecida")
    out["ferido"] = out["gravidade"].map(
        lambda g: np.nan if g == "Desconhecida" else g != "Sem feridos"
    ).astype("boolean")
    return out[COLUNAS_FINAIS]


def montar_populacao(pasta_brutos: Path = PASTA_BRUTOS) -> pd.DataFrame:
    partes = []
    for arquivo, (nome, _url) in ARQUIVOS_BRUTOS.items():
        bruto = ler_bruto(pasta_brutos / nome)
        partes.append(harmonizar(deduplicar(bruto), arquivo))
    pop = pd.concat(partes, ignore_index=True)
    if pop["id_incidente"].duplicated().any():
        raise ValueError("há acidentes repetidos entre os arquivos histórico e atual")
    return pop.sort_values("id_incidente", kind="stable").reset_index(drop=True)


def _tipar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["parado"] = df["parado"].astype(bool)
    df["ferido"] = df["ferido"].astype("boolean")
    df["id_incidente"] = df["id_incidente"].astype(str)
    df["ano"] = df["ano"].astype("Int64")
    df["hora"] = df["hora"].astype("Int64")
    return df


def carregar_populacao(caminho: Path = ARQ_POPULACAO) -> pd.DataFrame:
    return _tipar(pd.read_csv(caminho, dtype={"id_incidente": str, "id_relatorio": str}))


def carregar_amostra(caminho: Path = ARQ_AMOSTRA) -> pd.DataFrame:
    return _tipar(pd.read_csv(caminho, dtype={"id_incidente": str, "id_relatorio": str}))


def resumo_tratamento(pasta_brutos: Path = PASTA_BRUTOS) -> pd.DataFrame:
    """Contagens de cada etapa da limpeza, por arquivo bruto."""
    linhas = []
    for arquivo, (nome, url) in ARQUIVOS_BRUTOS.items():
        bruto = ler_bruto(pasta_brutos / nome)
        datas = pd.to_datetime(bruto["Incident Date"], format="%b-%Y", errors="coerce")
        linhas.append(
            {
                "arquivo": arquivo,
                "nome do arquivo": nome,
                "período dos acidentes": f"{datas.min():%m/%Y} a {datas.max():%m/%Y}",
                "linhas (versões de relatórios)": len(bruto),
                "colunas": bruto.shape[1],
                "relatórios distintos": bruto["Report ID"].nunique(),
                "acidentes distintos": len(deduplicar(bruto)),
                "url": url,
            }
        )
    return pd.DataFrame(linhas)
