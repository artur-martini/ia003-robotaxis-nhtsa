import math
from pathlib import Path

import pandas as pd
import streamlit as st

from analise import amostragem, dados, painel

pop = painel.populacao()
amo = painel.amostra()
plano = amostragem.plano_amostral(len(pop))
N, n = painel.inteiro(len(pop)), len(amo)


@st.cache_data
def resumo():
    return dados.resumo_tratamento()


r = resumo()
linhas_brutas = int(r["linhas (versões de relatórios)"].sum())

st.title("Acidentes com robotáxis: população × amostra")
painel.texto(
    f"Uma amostra aleatória de **{n} acidentes** consegue descrever os **{N} acidentes** com robotáxis informados "
    "ao governo dos EUA? Este trabalho compara as duas, com análise descritiva, intervalos de confiança e testes "
    "de hipóteses para proporções e médias."
)

paineis = [
    ("paginas/populacao.py", "Painel 1: População", f"Como são os {N} acidentes: indicadores, distribuições e frequências."),
    ("paginas/amostra.py", "Painel 2: Amostra", f"A mesma análise com os {n} acidentes sorteados, comparada com a população."),
    ("paginas/inferencia.py", "Painel 3: Inferência", "Quatro hipóteses testadas na amostra e conferidas com a população."),
]
for coluna, (pagina, titulo, descricao) in zip(st.columns(3), paineis):
    with coluna, st.container(border=True):
        st.markdown(f"**{titulo}**")
        st.caption(descricao)
        if st.button(f"Abrir o {titulo.split(':')[0]}", key=f"abrir_{pagina}"):
            st.switch_page(pagina)

# ---------------------------------------------------------------- base
st.header("1. A base de dados")

arquivos_html = " e ".join(painel.link(linha["nome do arquivo"], linha["url"]) for _, linha in r.iterrows())
painel.ficha(
    [
        ("Fonte", "NHTSA (National Highway Traffic Safety Administration), a agência de segurança viária do governo dos EUA."),
        ("Base", f"Standing General Order 2021-01, relatórios de incidentes com ADS ({painel.link('página oficial', dados.URL_PAGINA)})."),
        ("Arquivos originais", f"{arquivos_html}. Baixados em 15/11/2026 (atualização da NHTSA de 15/10/2026)."),
        (
            "Contexto e objetivo",
            "Desde 2021, a NHTSA obriga fabricantes e operadoras de veículos com <b>ADS</b> (sistema de direção "
            "automatizada, níveis 3 a 5 da SAE, como os robotáxis da Waymo, Zoox e Cruise) a informar todo acidente em "
            "que o sistema estava em uso ou tinha sido usado pouco antes. O objetivo é dar transparência e permitir "
            "que a agência identifique riscos de segurança.",
        ),
        ("Observações", f"{painel.inteiro(linhas_brutas)} linhas nos arquivos originais, que viram <b>{N} acidentes distintos</b> depois do tratamento."),
        ("Variáveis", "137 colunas no arquivo histórico e 116 no atual. O arquivo tratado tem 17, e a análise usa 12."),
        ("População de interesse", f"Todos os acidentes com ADS informados à NHTSA de julho de 2021 a setembro de 2026 (<b>N = {N}</b>)."),
        ("Unidade amostral", "O acidente."),
    ]
)
painel.texto(
    "A base é um registro administrativo completo, e não uma pesquisa por amostragem. Por isso os parâmetros da "
    "população podem ser calculados diretamente e comparados com o que a amostra estima."
)

st.subheader("Como os dados foram tratados")
painel.texto(
    f"""
Cada acidente pode aparecer em várias linhas: uma para cada atualização do relatório e uma para cada empresa
que o informou (Cruise e GM, por exemplo, informam o mesmo acidente). Para ficar com um registro por acidente:

1. Mantive só a **última versão** de cada relatório (`Report ID`).
2. Mantive **um relatório por acidente** (`Same Incident ID`). Quinze relatórios da Mobileye vieram sem esse
   campo, e cada um conta como um acidente próprio.
3. Juntei os dois arquivos. Nenhum acidente aparece nos dois.
4. Traduzi e unifiquei as categorias, porque o formulário mudou em 2025 e as escalas de gravidade são diferentes.
"""
)
tratamento = r[["arquivo", "período dos acidentes", "linhas (versões de relatórios)", "relatórios distintos", "acidentes distintos"]].copy()
tratamento.columns = ["Arquivo", "Período dos acidentes", "Linhas", "Relatórios distintos", "Acidentes distintos"]
total = {
    "Arquivo": "**Total**",
    "Período dos acidentes": "07/2021 a 09/2026",
    "Linhas": tratamento["Linhas"].sum(),
    "Relatórios distintos": tratamento["Relatórios distintos"].sum(),
    "Acidentes distintos": tratamento["Acidentes distintos"].sum(),
}
tratamento = pd.concat([tratamento, pd.DataFrame([total])], ignore_index=True)
for coluna in ("Linhas", "Relatórios distintos", "Acidentes distintos"):
    tratamento[coluna] = [painel.inteiro(v) for v in tratamento[coluna]]
tratamento.iloc[-1, 2:] = [f"**{v}**" for v in tratamento.iloc[-1, 2:]]
with st.container(width=980):
    st.table(tratamento.set_index("Arquivo"), border="horizontal")

# ---------------------------------------------------------------- variáveis
st.header("2. Variáveis da análise")
variaveis = pd.DataFrame(
    [
        ("`velocidade_mph`", "Quantitativa contínua", "Velocidade do robotáxi imediatamente antes da colisão, em mph (também em km/h)", "SV Precrash Speed (MPH)"),
        ("`hora`", "Quantitativa discreta", "Hora do dia do acidente (0 a 23)", "Incident Time (24:00)"),
        ("`ano`", "Quantitativa discreta", "Ano do acidente", "Incident Date"),
        ("`movimento`", "Qualitativa nominal", "O que o robotáxi fazia antes da colisão", "SV Pre-Crash Movement"),
        ("`parado`", "Qualitativa binária", "Robotáxi parado no trânsito ou estacionado (derivada de `movimento`)", "SV Pre-Crash Movement"),
        ("`gravidade`", "Qualitativa ordinal", "Lesão mais grave alegada: sem feridos, leve, moderada, grave ou fatal", "Highest Injury Severity Alleged"),
        ("`ferido`", "Qualitativa binária", "Houve ao menos um ferido (derivada de `gravidade`)", "Highest Injury Severity Alleged"),
        ("`colidiu_com`", "Qualitativa nominal", "Com o que o robotáxi colidiu: carro, SUV, ciclista, objeto fixo etc.", "Crash With"),
        ("`tipo_via`", "Qualitativa nominal", "Rua, cruzamento, estacionamento, rodovia etc.", "Roadway Type"),
        ("`empresa_grupo`", "Qualitativa nominal", "Empresa que informou o acidente: Waymo, Cruise (GM), Zoox etc.", "Reporting Entity"),
        ("`estado`", "Qualitativa nominal", "Estado dos EUA onde o acidente ocorreu", "State"),
        ("`arquivo`", "Qualitativa nominal", "Arquivo de origem: histórico ou atual", "(criada no tratamento)"),
    ],
    columns=["Variável", "Tipo", "O que mede", "Coluna original na NHTSA"],
)
st.table(variaveis.set_index("Variável"), border="horizontal")
painel.texto(
    "As hipóteses do Painel 3 usam `parado` e `ferido` para as proporções e `velocidade_mph` para as médias."
)

# ---------------------------------------------------------------- tamanho amostral
st.header("3. Tamanho da amostra")
painel.texto(
    "O tamanho foi calculado para **estimar uma proporção**, com a fórmula da calculadora de tamanho amostral "
    "indicada no enunciado (USP/Bauru). A proporção serve para dimensionar porque não exige conhecer a variância "
    "de antemão."
)
parametros = pd.DataFrame(
    [
        ("População (N)", N, "Acidentes distintos da base tratada."),
        ("Nível de confiança", f"95% (z = {painel.num(plano.z, 2)})", "Padrão da área, e o mesmo nível dos intervalos do Painel 3."),
        ("Margem de erro (e)", "5 pontos percentuais", "Precisão suficiente para separar proporções como 50% e 28%, testadas nas hipóteses."),
        ("Proporção esperada (p)", "50%", "Não há estimativa prévia. Com p = 0,5 a amostra é a maior possível: é a escolha conservadora."),
    ],
    columns=["Parâmetro", "Valor", "Justificativa"],
)
with st.container(width=980):
    st.table(parametros.set_index("Parâmetro"), border="horizontal")
    c1, c2 = st.columns([3, 2], gap="large", vertical_alignment="center")
    with c1:
        st.latex(r"n_0 = \frac{z^2\,p\,(1-p)}{e^2} = \frac{1{,}96^2 \times 0{,}5 \times 0{,}5}{0{,}05^2} \approx " + painel.tex(plano.n0, 2))
        st.latex(
            r"n = \frac{n_0}{1 + (n_0 - 1)/N} = \frac{"
            + painel.tex(plano.n0, 2)
            + r"}{1 + "
            + painel.tex(plano.n0 - 1, 2)
            + r"/"
            + N
            + r"} \approx "
            + painel.tex(plano.n_exato, 2)
        )
    with c2:
        painel.destaque(
            f"**Amostra de n = {plano.n} acidentes**, ou {painel.pct(plano.n / len(pop))} da população. "
            f"Sem a correção para população finita (a segunda fórmula), seriam {math.ceil(plano.n0)}.",
            ":material/calculate:",
        )
print_calc = Path(__file__).resolve().parent.parent / "referencias" / "calculadora_tamanho_amostral.png"
if print_calc.exists():
    st.image(str(print_calc), caption="Resultado na calculadora de tamanho amostral (USP/Bauru)", width=520)

# ---------------------------------------------------------------- AAS
st.header("4. Sorteio da amostra aleatória simples")
painel.texto(
    f"""
1. Ordenei os {N} acidentes pelo identificador (`id_incidente`), para que a ordem não dependa do arquivo de origem.
2. Sorteei **{plano.n} posições distintas, sem reposição**, entre 0 e {painel.inteiro(len(pop) - 1)}, com o gerador de
   números aleatórios do NumPy e a **semente fixa {amostragem.SEMENTE}**.
3. Os acidentes dessas posições formam a amostra.

Cada acidente tem a mesma probabilidade de entrar ({plano.n}/{N} ≈ {painel.pct(plano.n / len(pop))}), e todo
conjunto de {plano.n} acidentes tem a mesma chance de ser o sorteado. A semente torna o sorteio reproduzível:
quem rodar o código obtém a mesma amostra.
"""
)
with st.container(width=painel.LARGURA_TEXTO):
    st.code(
        f"""rng = np.random.default_rng({amostragem.SEMENTE})
base = populacao.sort_values("id_incidente").reset_index(drop=True)
posicoes = np.sort(rng.choice(len(base), size={plano.n}, replace=False))
amostra = base.iloc[posicoes]""",
        language="python",
    )
    c1, c2 = st.columns(2)
    c1.download_button("Baixar a população tratada (CSV)", dados.ARQ_POPULACAO.read_bytes(), "populacao_ads.csv", "text/csv", width="stretch", icon=":material/download:")
    c2.download_button("Baixar a amostra sorteada (CSV)", dados.ARQ_AMOSTRA.read_bytes(), "amostra_ads.csv", "text/csv", width="stretch", icon=":material/download:")

painel.rodape()
