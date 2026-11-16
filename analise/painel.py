"""Peças de Streamlit compartilhadas pelos painéis."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from analise import dados
from analise import estatistica as est
from analise import graficos as gr

LARGURA_TEXTO = 760  # largura de leitura dos parágrafos, em pixels

QUANTITATIVAS = {
    "velocidade_mph": {"rotulo": "Velocidade antes da colisão (mph)", "unidade": "mph", "classe": 5},
    "velocidade_kmh": {"rotulo": "Velocidade antes da colisão (km/h)", "unidade": "km/h", "classe": 8},
    "hora": {"rotulo": "Hora do dia", "unidade": "h", "classe": 1},
}

QUALITATIVAS = {
    "parado_txt": {"rotulo": "Robotáxi parado no momento da colisão?", "ordem": ["Sim", "Não"]},
    "ferido_txt": {"rotulo": "Houve feridos?", "ordem": ["Sim", "Não", "Desconhecido"]},
    "gravidade": {"rotulo": "Gravidade da lesão mais séria", "ordem": dados.ORDEM_GRAVIDADE},
    "movimento": {"rotulo": "O que o robotáxi fazia antes da colisão", "ordem": None},
    "colidiu_com": {"rotulo": "Com o que o robotáxi colidiu", "ordem": None},
    "tipo_via": {"rotulo": "Tipo de via", "ordem": None},
    "empresa_grupo": {"rotulo": "Empresa que informou o acidente", "ordem": dados.ORDEM_EMPRESA},
    "estado": {"rotulo": "Estado dos EUA", "ordem": None},
    "arquivo": {"rotulo": "Arquivo de origem na NHTSA", "ordem": list(dados.ARQUIVOS_BRUTOS)},
}

ESTADOS = {
    "CA": "na Califórnia",
    "TX": "no Texas",
    "AZ": "no Arizona",
    "GA": "na Geórgia",
    "NV": "em Nevada",
    "FL": "na Flórida",
}

NOMES_MEDIDAS = {
    "n": "n (válidos)",
    "faltantes": "faltantes",
    "media": "média",
    "mediana": "mediana",
    "moda": "moda",
    "desvio_padrao": "desvio-padrão",
    "variancia": "variância",
    "coef_variacao": "coeficiente de variação",
    "minimo": "mínimo",
    "q1": "1º quartil (Q1)",
    "q3": "3º quartil (Q3)",
    "maximo": "máximo",
    "amplitude_interquartil": "amplitude interquartil",
    "assimetria": "assimetria",
}


# ---------------------------------------------------------------- formatação


def _bruto(v: float, casas: int) -> str:
    return f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def num(v: float, casas: int = 2) -> str:
    """Número no padrão brasileiro, com o sinal de menos tipográfico."""
    if v is None or pd.isna(v):
        return "—"
    return _bruto(v, casas).replace("-", "−")


def tex(v: float, casas: int = 2) -> str:
    """Número com vírgula decimal para usar dentro de st.latex."""
    return _bruto(v, casas).replace(",", "{,}")


def inteiro(v: int) -> str:
    return f"{int(v):,}".replace(",", ".")


def pct(v: float, casas: int = 1) -> str:
    if v is None or pd.isna(v):
        return "—"
    return num(v * 100, casas) + "%"


def valor_p(v: float) -> str:
    if v < 0.0001:
        return "< 0,0001"
    return num(v, 4)


# ---------------------------------------------------------------- dados


def _com_textos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["parado_txt"] = df["parado"].map({True: "Sim", False: "Não"})
    df["ferido_txt"] = df["ferido"].map({True: "Sim", False: "Não"}).fillna("Desconhecido")
    return df


@st.cache_data
def populacao() -> pd.DataFrame:
    return _com_textos(dados.carregar_populacao())


@st.cache_data
def amostra() -> pd.DataFrame:
    return _com_textos(dados.carregar_amostra())


# ---------------------------------------------------------------- peças de layout


def texto(markdown: str) -> None:
    """Parágrafo com largura de leitura limitada (o layout da página é largo, por causa dos gráficos)."""
    with st.container(width=LARGURA_TEXTO):
        st.markdown(markdown)


def grafico(fig, titulo: str | None = None, key: str | None = None, nota: str | None = None) -> None:
    """Gráfico Plotly com título e nota escritos pelo Streamlit, e sem a barra de ferramentas."""
    if titulo:
        st.markdown(f"**{titulo}**")
    st.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": False})
    if nota:
        st.caption(nota)


def faixa_metricas(itens: list[tuple], key: str | None = None, larguras: list[float] | None = None) -> None:
    """Uma faixa única de indicadores. Cada item é (rótulo, valor) ou (rótulo, valor, complemento)."""
    with st.container(border=True, key=key):
        colunas = st.columns(larguras or len(itens))
        for coluna, item in zip(colunas, itens):
            rotulo, valor = item[0], item[1]
            complemento = item[2] if len(item) > 2 else None
            with coluna:
                st.metric(rotulo, valor)
                if complemento:
                    st.caption(complemento)


def destaque(markdown: str, icone: str = ":material/lightbulb:", largura: int | str = "stretch") -> None:
    """Bloco de leitura dos resultados, em tinta neutra (azul e laranja ficam reservados aos dados)."""
    with st.container(border=True, width=largura):
        st.markdown(f"{icone} {markdown}")


def ficha(itens: list[tuple[str, str]], largura_rem: int = 62) -> None:
    """Lista de rótulo e conteúdo (o conteúdo aceita HTML simples: <b>, <a>, <code>)."""
    linha = "display:grid;grid-template-columns:11.5rem 1fr;gap:1.25rem;padding:0.6rem 0;border-top:1px solid #d3d8d5"
    corpo = "".join(
        f'<div style="{linha}"><div style="color:{gr.TINTA_SUAVE}">{rotulo}</div><div>{conteudo}</div></div>'
        for rotulo, conteudo in itens
    )
    st.html(f'<div style="max-width:{largura_rem}rem;line-height:1.55;border-bottom:1px solid #d3d8d5">{corpo}</div>')


def link(texto: str, url: str) -> str:
    return f'<a href="{url}" target="_blank" style="color:#1f5fae">{texto}</a>'


def legenda_cores(n_pop: int, n_amo: int) -> str:
    """Legenda fixa das cores, para a barra lateral."""
    linha = "display:flex;gap:0.55rem;align-items:baseline;margin:0.3rem 0"
    return f"""
<div style="font-size:0.85rem;line-height:1.35">
  <div style="font-weight:700;margin-bottom:0.4rem">Como ler as cores</div>
  <div style="{linha}"><span style="color:{gr.AZUL}">●</span><span><b>População</b>: os {inteiro(n_pop)} acidentes, valores reais</span></div>
  <div style="{linha}"><span style="color:{gr.LARANJA}">●</span><span><b>Amostra</b>: {n_amo} acidentes sorteados, estimativas</span></div>
  <div style="{linha}"><span style="color:#aeb4b0">┆</span><span><b>H₀</b>: valor testado na hipótese nula</span></div>
</div>
"""


def rodape() -> None:
    st.divider()
    st.caption(
        "Trabalho final de IA003 (Estatística), por Artur Martini da Rosa. "
        f"Dados: NHTSA, Standing General Order 2021-01, incidentes com ADS ([página oficial]({dados.URL_PAGINA}))."
    )


# ---------------------------------------------------------------- painel descritivo


def _filtros(chave: str, base: pd.DataFrame) -> tuple[tuple[int, int], list[str], bool]:
    anos = base["ano"].dropna().astype(int)
    c1, c2 = st.columns([1, 2])
    with c1:
        periodo = st.slider("Ano do acidente", int(anos.min()), int(anos.max()), (int(anos.min()), int(anos.max())), key=f"{chave}_anos")
    with c2:
        empresas = st.multiselect(
            "Empresa que informou o acidente",
            dados.ORDEM_EMPRESA,
            default=dados.ORDEM_EMPRESA,
            key=f"{chave}_empresas",
            help="A Waymo concentra cerca de três quartos dos acidentes.",
        )
    periodo_completo = periodo == (int(anos.min()), int(anos.max()))
    return periodo, empresas, periodo_completo


def _aplicar(df: pd.DataFrame, periodo, empresas, periodo_completo: bool) -> pd.DataFrame:
    """Filtra por ano e empresa. O acidente sem data só fica quando o período inteiro está selecionado."""
    ano = df["ano"]
    no_periodo = ano.between(*periodo).fillna(False).astype(bool)
    if periodo_completo:
        no_periodo = no_periodo | ano.isna()
    return df[no_periodo & df["empresa_grupo"].isin(empresas)]


def _kpis(df: pd.DataFrame, ref: pd.DataFrame | None, chave: str) -> None:
    v = df["velocidade_mph"].dropna()
    ferido = df["ferido"].dropna().astype(bool)

    def comp(texto_ref: str) -> str | None:
        return None if ref is None else f"população: {texto_ref}"

    vr = ref["velocidade_mph"] if ref is not None else None
    itens = [
        ("Acidentes", inteiro(len(df)), comp(inteiro(len(ref))) if ref is not None else None),
        ("Robotáxi parado", pct(df["parado"].mean()), comp(pct(ref["parado"].mean())) if ref is not None else None),
        ("Com feridos", pct(ferido.mean()), comp(pct(ref["ferido"].dropna().astype(bool).mean())) if ref is not None else None),
        ("Velocidade média", num(v.mean(), 1) + " mph", comp(num(vr.mean(), 1) + " mph") if ref is not None else None),
        ("Velocidade mediana", num(v.median(), 1) + " mph", comp(num(vr.median(), 1) + " mph") if ref is not None else None),
        ("Desvio-padrão", num(v.std(), 1) + " mph", comp(num(vr.std(), 1) + " mph") if ref is not None else None),
    ]
    faixa_metricas(itens, key=f"{chave}_kpis")


def leitura_rapida(df: pd.DataFrame) -> list[str]:
    """Frases que resumem o recorte, calculadas dos próprios dados."""
    v = df["velocidade_mph"].dropna()
    ferido = df["ferido"].dropna().astype(bool)
    empresas = df["empresa_grupo"].value_counts(normalize=True)
    estados = df["estado"].value_counts(normalize=True)
    fatais = int((df["gravidade"] == "Fatal").sum())
    frases = [
        f"**Em {pct(df['parado'].mean())} dos acidentes o robotáxi estava parado** (no trânsito ou estacionado): "
        "o outro envolvido é que estava em movimento.",
        f"**A velocidade é baixa e muito assimétrica**: mediana de {num(v.median(), 0)} mph e média de "
        f"{num(v.mean(), 1)} mph ({num(v.mean() * dados.MPH_PARA_KMH, 1)} km/h). "
        f"Só {pct((v > 25).mean())} dos acidentes ocorreram acima de 25 mph.",
        f"**{pct(ferido.mean())} dos acidentes tiveram feridos**, entre os de gravidade conhecida. "
        + ("Nenhum foi fatal." if fatais == 0 else f"{fatais} {'foi fatal' if fatais == 1 else 'foram fatais'}."),
    ]
    if len(empresas) and len(estados):
        empresa, uf = empresas.index[0], estados.index[0]
        por_quem = "por empresas menores" if empresa == "Outras" else f"pela {empresa}"
        frases.append(
            f"**Os dados são concentrados**: {pct(empresas.iloc[0])} dos acidentes foram informados {por_quem}, "
            f"e {pct(estados.iloc[0])} ocorreram {ESTADOS.get(uf, f'no estado {uf}')}."
        )
    return frases


def tabela_medidas(series: dict[str, pd.Series], unidade: str) -> pd.DataFrame:
    linhas = {nome: est.descritivas(s) for nome, s in series.items()}
    t = pd.DataFrame(linhas)
    t.index = [NOMES_MEDIDAS[i] for i in t.index]
    fmt = t.copy().astype(object)
    for i in t.index:
        for c in t.columns:
            v = t.loc[i, c]
            if i in ("n (válidos)", "faltantes"):
                fmt.loc[i, c] = inteiro(v)
            elif i in ("coeficiente de variação", "assimetria"):
                fmt.loc[i, c] = num(v, 2)
            elif i == "variância":
                fmt.loc[i, c] = num(v, 2) + f" {unidade}²"
            else:
                fmt.loc[i, c] = num(v, 2) + f" {unidade}"
    if len(series) == 2:
        a, b = t.columns
        dif = (t[b] - t[a]).astype(float)
        fmt["Diferença"] = ["—" if i in ("n (válidos)", "faltantes") else num(d, 2) for i, d in zip(t.index, dif)]
    return fmt.rename_axis("Medida").reset_index()


def tabela_frequencias(series: dict[str, pd.Series], ordem) -> pd.DataFrame:
    partes = []
    for nome, s in series.items():
        f = est.frequencias(s.astype(str), ordem).set_index("categoria")
        f.columns = [f"{nome} (n)", f"{nome} (%)"]
        partes.append(f)
    t = pd.concat(partes, axis=1).fillna(0)
    saida = pd.DataFrame(index=t.index)
    for c in t.columns:
        saida[c] = [inteiro(v) for v in t[c]] if c.endswith("(n)") else [num(v * 100, 1) for v in t[c]]
    if len(series) == 2:
        a, b = series
        saida["Diferença (p.p.)"] = [num(d * 100, 1) for d in t[f"{b} (%)"] - t[f"{a} (%)"]]
    return saida.rename_axis("Categoria").reset_index()


def painel_descritivo(df_base: pd.DataFrame, nome: str, referencia: pd.DataFrame | None = None) -> None:
    chave = nome.lower()
    periodo, empresas, completo = _filtros(chave, df_base if referencia is None else referencia)
    df = _aplicar(df_base, periodo, empresas, completo)
    ref = _aplicar(referencia, periodo, empresas, completo) if referencia is not None else None
    if len(df) < len(df_base):
        st.caption(f"Filtro ativo: {inteiro(len(df))} de {inteiro(len(df_base))} acidentes. Limpe os filtros para ver todos.")
    if df.empty:
        st.warning("Nenhum acidente com esses filtros. Amplie o período ou inclua mais empresas.")
        return

    comparar = False
    if ref is not None:
        comparar = st.toggle("Comparar com a população nos gráficos", value=True, key=f"{chave}_comparar")

    def series(col: str) -> dict[str, pd.Series]:
        if comparar:
            return {"População": ref[col], nome: df[col]}
        return {nome: df[col]}

    _kpis(df, ref if comparar else None, chave)

    with st.container(width=LARGURA_TEXTO):
        st.markdown(f"**O que os dados da {nome.lower()} mostram**")
        st.markdown("\n".join(f"- {frase}" for frase in leitura_rapida(df)))

    abas = st.tabs(["Visão geral", "Velocidade e hora (quantitativas)", "Categorias (qualitativas)"])

    with abas[0]:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            grafico(gr.acidentes_por_ano(series("ano")), "Acidentes por ano (% do total)", key=f"{chave}_ano")
        with c2:
            grafico(gr.mapa_estados(df["estado"], nome), f"Onde aconteceram: estados dos EUA ({nome.lower()})", key=f"{chave}_mapa")
        c3, c4 = st.columns(2, gap="large")
        with c3:
            grafico(
                gr.barras_frequencia(series("movimento")),
                "O que o robotáxi fazia antes da colisão",
                key=f"{chave}_movimento",
                nota="Robotáxi parado = \"Parado no trânsito\" ou \"Estacionado\".",
            )
        with c4:
            grafico(
                gr.barras_frequencia(series("gravidade"), ordem=dados.ORDEM_GRAVIDADE),
                "Gravidade da lesão mais séria",
                key=f"{chave}_gravidade",
                nota="Lesão mais grave alegada entre todos os envolvidos, não só os ocupantes do robotáxi.",
            )

    with abas[1]:
        var = st.segmented_control(
            "Variável",
            list(QUANTITATIVAS),
            format_func=lambda k: QUANTITATIVAS[k]["rotulo"],
            default="velocidade_mph",
            key=f"{chave}_quanti",
        ) or "velocidade_mph"
        meta = QUANTITATIVAS[var]
        c1, c2 = st.columns([1, 1] if comparar else [3, 2], gap="large")
        with c1:
            grafico(
                gr.histograma(series(var), meta["unidade"], meta["classe"]),
                f"Distribuição: {meta['rotulo'].lower()}",
                key=f"{chave}_histograma",
            )
            grafico(gr.boxplot(series(var), meta["unidade"]), "Boxplot (a linha tracejada marca a média)", key=f"{chave}_boxplot")
        with c2:
            st.markdown("**Medidas descritivas**")
            st.table(tabela_medidas(series(var), meta["unidade"]).set_index("Medida"), border="horizontal")
        if var.startswith("velocidade"):
            destaque(
                "**Por que a mediana é 0?** Em mais da metade dos acidentes o robotáxi estava parado, com velocidade 0. "
                "A média fica acima da mediana porque poucos acidentes em alta velocidade a puxam para cima "
                "(assimetria à direita).",
                largura=LARGURA_TEXTO + 60,
            )

    with abas[2]:
        var = st.selectbox(
            "Variável", list(QUALITATIVAS), format_func=lambda k: QUALITATIVAS[k]["rotulo"], key=f"{chave}_quali", width=420
        )
        meta = QUALITATIVAS[var]
        # com duas séries a tabela tem seis colunas e não cabe ao lado do gráfico: vai embaixo
        c1, c2 = st.columns([3, 2], gap="large") if not comparar else (st.container(width=980), st.container(width=980))
        with c1:
            grafico(gr.barras_frequencia(series(var), ordem=meta["ordem"]), f"{meta['rotulo']} (% dos acidentes)", key=f"{chave}_barras_quali")
        with c2:
            st.markdown("**Frequências e proporções**")
            st.dataframe(tabela_frequencias(series(var), meta["ordem"]), hide_index=True, width="stretch", key=f"{chave}_tabela_freq")
