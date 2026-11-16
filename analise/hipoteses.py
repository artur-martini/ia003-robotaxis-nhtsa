"""As quatro hipóteses do trabalho: definição, cálculo na amostra e comparação com a população.

O nível de significância e o nível de confiança são fixados aqui, antes de
qualquer cálculo com a amostra.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analise import estatistica as est

ALFA = 0.05
CONFIANCA = 0.95


@dataclass(frozen=True)
class Referencia:
    texto: str
    url: str


@dataclass(frozen=True)
class Hipotese:
    codigo: str
    tipo: str  # "proporcao", "media" ou "duas_medias"
    titulo: str
    pergunta: str
    variavel: str
    evento: str
    parametro: str  # símbolo do parâmetro, ex.: "π"
    valor_h0: float
    alternativa: str
    h0: str
    h1: str
    justificativa: str
    referencias: tuple[Referencia, ...] = field(default_factory=tuple)
    unidade: str = ""
    pergunta_curta: str = ""


REF_NHTSA_2023 = Referencia(
    "NHTSA (2025), Summary of Motor Vehicle Traffic Crashes: 2023 Data (DOT HS 813 762), Tabela 1",
    "https://crashstats.nhtsa.dot.gov/Api/Public/ViewPublication/813762",
)
REF_CVC_22352 = Referencia(
    "California Vehicle Code §22352 (limite de 25 mph em áreas comerciais e residenciais)",
    "https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=VEH&sectionNum=22352",
)

HIPOTESES: dict[str, Hipotese] = {
    "P1": Hipotese(
        codigo="P1",
        tipo="proporcao",
        titulo="Robotáxi parado no momento da colisão",
        pergunta="A maioria dos acidentes acontece com o robotáxi parado (no trânsito ou estacionado)?",
        pergunta_curta="A maioria dos acidentes acontece com o robotáxi parado?",
        variavel="parado",
        evento="robotáxi parado no trânsito ou estacionado antes da colisão",
        parametro="π",
        valor_h0=0.5,
        alternativa="maior",
        h0="π = 0,50 (metade dos acidentes ocorre com o robotáxi parado)",
        h1="π > 0,50 (a maioria ocorre com o robotáxi parado)",
        justificativa=(
            "0,50 é o limite de \"maioria\". Se o robotáxi está parado quando a batida acontece, o outro envolvido "
            "é que estava em movimento. Uma maioria de acidentes assim indicaria que boa parte deles não parte de "
            "uma manobra do sistema autônomo."
        ),
    ),
    "P2": Hipotese(
        codigo="P2",
        tipo="proporcao",
        titulo="Acidentes com feridos",
        pergunta="Os acidentes com robotáxi deixam feridos com menos frequência que os acidentes de trânsito em geral nos EUA (28%)?",
        pergunta_curta="Há menos acidentes com feridos do que no trânsito em geral?",
        variavel="ferido",
        evento="acidente com lesão leve, moderada, grave ou fatal (os de gravidade desconhecida ficam de fora)",
        parametro="π",
        valor_h0=0.28,
        alternativa="menor",
        h0="π = 0,28 (mesma parcela com feridos dos acidentes em geral)",
        h1="π < 0,28 (parcela menor de acidentes com feridos)",
        justificativa=(
            "Entre os acidentes registrados pela polícia nos EUA em 2023, 27,6% tiveram feridos e 0,6% tiveram "
            "mortos, 28,2% no total. O valor vem dessa estatística nacional, e não dos dados do trabalho."
        ),
        referencias=(REF_NHTSA_2023,),
    ),
    "M1": Hipotese(
        codigo="M1",
        tipo="media",
        titulo="Velocidade do robotáxi antes da colisão",
        pergunta="A velocidade média do robotáxi antes da colisão fica abaixo do limite urbano de 25 mph (40 km/h)?",
        pergunta_curta="A velocidade média na colisão fica abaixo de 25 mph?",
        variavel="velocidade_mph",
        evento="velocidade do robotáxi imediatamente antes da colisão, em mph",
        parametro="μ",
        valor_h0=25.0,
        alternativa="menor",
        h0="μ = 25 mph (velocidade média igual ao limite urbano)",
        h1="μ < 25 mph (velocidade média abaixo do limite urbano)",
        justificativa=(
            "25 mph (40 km/h) é o limite de velocidade padrão em áreas comerciais e residenciais da Califórnia, "
            "estado com mais da metade dos acidentes da base. É uma referência de velocidade urbana."
        ),
        referencias=(REF_CVC_22352,),
        unidade="mph",
    ),
    "M2": Hipotese(
        codigo="M2",
        tipo="duas_medias",
        titulo="Velocidade nos acidentes com e sem feridos",
        pergunta="Nos acidentes com feridos, o robotáxi andava mais rápido, em média, do que nos acidentes sem feridos?",
        pergunta_curta="O robotáxi andava mais rápido nos acidentes com feridos?",
        variavel="velocidade_mph",
        evento="velocidade do robotáxi antes da colisão, em mph, nos acidentes com feridos (grupo 1) e sem feridos (grupo 2)",
        parametro="μ₁ − μ₂",
        valor_h0=0.0,
        alternativa="maior",
        h0="μ₁ = μ₂ (mesma velocidade média nos dois grupos)",
        h1="μ₁ > μ₂ (velocidade média maior nos acidentes com feridos)",
        justificativa=(
            "Mais velocidade significa mais energia na colisão e mais risco de lesão. Se a velocidade do robotáxi "
            "pesasse na gravidade, ela seria maior nos acidentes com feridos. H₀ supõe que não há diferença."
        ),
        unidade="mph",
    ),
}


def dados_da_hipotese(h: Hipotese, df: pd.DataFrame):
    """Devolve os dados usados por uma hipótese (sem faltantes)."""
    if h.tipo == "proporcao":
        s = df[h.variavel].dropna().astype(bool)
        return int(s.sum()), int(s.size)
    if h.tipo == "media":
        return df[h.variavel].dropna().to_numpy(dtype=float)
    base = df.dropna(subset=["ferido", h.variavel])
    g1 = base.loc[base["ferido"].astype(bool), h.variavel].to_numpy(dtype=float)
    g2 = base.loc[~base["ferido"].astype(bool), h.variavel].to_numpy(dtype=float)
    return g1, g2


def parametro_populacional(h: Hipotese, populacao: pd.DataFrame) -> float:
    d = dados_da_hipotese(h, populacao)
    if h.tipo == "proporcao":
        x, n = d
        return x / n
    if h.tipo == "media":
        return float(np.mean(d))
    g1, g2 = d
    return float(np.mean(g1) - np.mean(g2))


@dataclass(frozen=True)
class Avaliacao:
    hipotese: Hipotese
    teste: est.ResultadoTeste
    ic: est.Intervalo
    parametro_pop: float
    n_pop: int
    detalhes: dict

    @property
    def h0_verdadeira_na_populacao(self) -> bool:
        """Na população, H₀ (no sentido da alternativa) é verdadeira quando o parâmetro não está do lado de H₁."""
        h = self.hipotese
        if h.alternativa == "menor":
            return not self.parametro_pop < h.valor_h0
        if h.alternativa == "maior":
            return not self.parametro_pop > h.valor_h0
        return self.parametro_pop == h.valor_h0

    @property
    def tipo_de_decisao(self) -> str:
        rejeitou = self.teste.rejeita_h0
        h0_vale = self.h0_verdadeira_na_populacao
        if rejeitou and not h0_vale:
            return "Decisão correta (rejeitou H₀, que é falsa na população)"
        if not rejeitou and h0_vale:
            return "Decisão correta (não rejeitou H₀, que é verdadeira na população)"
        if rejeitou and h0_vale:
            return "Erro tipo I (rejeitou H₀, que é verdadeira na população)"
        return "Erro tipo II (não rejeitou H₀, que é falsa na população)"

    @property
    def ic_contem_parametro(self) -> bool:
        return self.ic.contem(self.parametro_pop)


def avaliar(h: Hipotese, amostra: pd.DataFrame, populacao: pd.DataFrame, alfa: float = ALFA, confianca: float = CONFIANCA) -> Avaliacao:
    d = dados_da_hipotese(h, amostra)
    dp = dados_da_hipotese(h, populacao)
    detalhes: dict = {}
    if h.tipo == "proporcao":
        x, n = d
        teste = est.teste_proporcao(x, n, h.valor_h0, h.alternativa, alfa)
        ic = est.ic_proporcao(x, n, confianca)
        n_pop = dp[1]
        detalhes = {
            "x": x,
            "n": n,
            "n_pi0": n * h.valor_h0,
            "n_1_menos_pi0": n * (1 - h.valor_h0),
            "x_pop": dp[0],
        }
    elif h.tipo == "media":
        teste = est.teste_t_uma_amostra(d, h.valor_h0, h.alternativa, alfa)
        ic = est.ic_media(d, confianca)
        n_pop = len(dp)
        detalhes = {"n": len(d), "desvio_padrao": float(np.std(d, ddof=1)), "assimetria": est.descritivas(pd.Series(d))["assimetria"]}
    else:
        g1, g2 = d
        teste = est.teste_welch(g1, g2, h.alternativa, alfa)
        ic = est.ic_diferenca_medias_welch(g1, g2, confianca)
        n_pop = len(dp[0]) + len(dp[1])
        detalhes = {
            "n1": len(g1),
            "n2": len(g2),
            "media1": float(np.mean(g1)),
            "media2": float(np.mean(g2)),
            "dp1": float(np.std(g1, ddof=1)),
            "dp2": float(np.std(g2, ddof=1)),
            "media1_pop": float(np.mean(dp[0])),
            "media2_pop": float(np.mean(dp[1])),
            "n1_pop": len(dp[0]),
            "n2_pop": len(dp[1]),
        }
    return Avaliacao(h, teste, ic, parametro_populacional(h, populacao), n_pop, detalhes)


def avaliar_todas(amostra: pd.DataFrame, populacao: pd.DataFrame) -> dict[str, Avaliacao]:
    return {c: avaliar(h, amostra, populacao) for c, h in HIPOTESES.items()}


def simular_amostras(h: Hipotese, populacao: pd.DataFrame, n: int, repeticoes: int, semente_inicial: int = 1) -> pd.DataFrame:
    """Repete a AAS muitas vezes e guarda, para cada amostra, a estimativa, o IC e o valor-p."""
    from analise.amostragem import sortear_aas

    linhas = []
    for i in range(repeticoes):
        a = sortear_aas(populacao, n, semente_inicial + i)
        r = avaliar(h, a, populacao)
        linhas.append(
            {
                "amostra": i + 1,
                "estimativa": r.ic.estimativa,
                "inferior": r.ic.inferior,
                "superior": r.ic.superior,
                "contem": r.ic_contem_parametro,
                "valor_p": r.teste.valor_p,
                "rejeita": r.teste.rejeita_h0,
            }
        )
    return pd.DataFrame(linhas)
