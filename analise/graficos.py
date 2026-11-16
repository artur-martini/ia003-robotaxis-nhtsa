"""Gráficos Plotly dos painéis.

Cores fixas por papel (paleta categórica validada para daltonismo):
população = azul, amostra = laranja, valor de H₀ = cinza.
Os títulos dos gráficos são escritos pelo Streamlit, fora da figura.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from analise import estatistica as est

AZUL = "#2a78d6"  # população
LARANJA = "#eb6834"  # amostra
CINZA = "#898781"  # referência / H₀
CINZA_CLARO = "#c5c9c6"
TINTA = "#1c2430"
TINTA_SUAVE = "#566170"
GRADE = "#e6e9e7"
ESCALA_AZUL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ESCALA_LARANJA = ["#fde3d6", "#f9c0a5", "#f39a74", "#eb6834", "#c9501f", "#9c3c15", "#6e290d"]

COR = {"População": AZUL, "Amostra": LARANJA}
FONTE = '"Public Sans", system-ui, -apple-system, "Segoe UI", sans-serif'
TRANSPARENTE = "rgba(0,0,0,0)"


def _pct(v: float, casas: int = 1) -> str:
    return f"{v * 100:.{casas}f}%".replace(".", ",")


def _num(v: float, casas: int = 2) -> str:
    return f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".").replace("-", "−")


def tema(fig: go.Figure, altura: int = 340, legenda: bool = False, topo: int | None = None) -> go.Figure:
    fig.update_layout(
        height=altura,
        margin=dict(l=8, r=8, t=topo if topo is not None else (36 if legenda else 12), b=8),
        title=None,
        font=dict(family=FONTE, size=13, color=TINTA_SUAVE),
        showlegend=legenda,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title=None, font=dict(color=TINTA)),
        separators=",.",
        paper_bgcolor=TRANSPARENTE,
        plot_bgcolor=TRANSPARENTE,
        hoverlabel=dict(font=dict(family=FONTE)),
        bargap=0.25,
        bargroupgap=0.08,
    )
    fig.update_xaxes(showline=False, zeroline=False, ticks="", gridcolor=GRADE)
    fig.update_yaxes(showline=False, zeroline=False, ticks="", gridcolor=GRADE)
    return fig


# ---------------------------------------------------------------- descritiva


def barras_frequencia(series: dict[str, pd.Series], ordem: list | None = None, max_categorias: int = 12) -> go.Figure:
    """Barras horizontais de proporção. `series` mapeia "População"/"Amostra" para a variável.

    As categorias seguem a ordem dada ou, sem ordem, a frequência na primeira série. Sem ordem fixa, as
    categorias além das `max_categorias - 1` mais frequentes são somadas em "Demais", em todas as séries.
    """
    presentes = set(pd.concat(list(series.values())).astype(str))
    principal = est.frequencias(next(iter(series.values())).astype(str), ordem)
    categorias = principal["categoria"].tolist()
    if ordem is not None:
        categorias = [c for c in ordem if c in presentes]
    else:
        todas = pd.concat(list(series.values())).astype(str).value_counts().index.tolist()
        categorias = categorias + [c for c in todas if c not in categorias]
        if len(categorias) > max_categorias:
            categorias = categorias[: max_categorias - 1] + ["Demais"]
    fig = go.Figure()
    for nome, s in series.items():
        s = s.astype(str)
        if "Demais" in categorias:
            s = s.where(s.isin(categorias), "Demais")
        f = est.frequencias(s).set_index("categoria").reindex(categorias).fillna(0)
        fig.add_bar(
            name=nome,
            y=f.index,
            x=f["proporcao"],
            orientation="h",
            marker=dict(color=COR.get(nome, AZUL), line=dict(width=0)),
            text=[_pct(v) for v in f["proporcao"]],
            textposition="outside",
            textfont=dict(color=TINTA, size=12),
            cliponaxis=False,
            customdata=f["frequencia"].astype(int),
            hovertemplate="%{y}<br>" + nome + ": %{x:.1%} (%{customdata} acidentes)<extra></extra>",
        )
    n_cat = len(categorias)
    comparando = len(series) > 1
    fig.update_yaxes(autorange="reversed", tickfont=dict(color=TINTA))
    fig.update_xaxes(tickformat=".0%", showgrid=True, range=[0, max(max(t.x) for t in fig.data) * 1.2])
    fig.update_layout(barmode="group")
    altura = max(190, (40 if comparando else 30) * n_cat + (80 if comparando else 50))
    return tema(fig, altura=altura, legenda=comparando)


def histograma(series: dict[str, pd.Series], rotulo_x: str, largura_classe: float, inicio: float = 0) -> go.Figure:
    comparando = len(series) > 1
    fig = go.Figure()
    for nome, s in series.items():
        fig.add_histogram(
            name=nome,
            x=s.dropna(),
            histnorm="percent",
            xbins=dict(start=inicio, size=largura_classe),
            marker=dict(color=COR.get(nome, AZUL), line=dict(width=1, color="white")),
            hovertemplate=nome + "<br>" + rotulo_x + ": %{x}<br>%{y:.1f}% dos acidentes<extra></extra>",
        )
    if not comparando:
        s = next(iter(series.values())).dropna()
        # folga acima da barra mais alta, para os rótulos de média e mediana não cobrirem os dados
        classes = np.arange(inicio, float(s.max()) + 2 * largura_classe, largura_classe)
        maior_barra = np.histogram(s, bins=classes)[0].max() / len(s) * 100
        fig.update_yaxes(range=[0, maior_barra * 1.32])
        mediana, media = float(s.median()), float(s.mean())
        fig.add_vline(x=mediana, line=dict(color=TINTA, width=1.5, dash="dot"))
        fig.add_vline(x=media, line=dict(color=TINTA, width=1.5, dash="dash"))
        fig.add_annotation(
            x=max(mediana, media),
            y=0.99,
            yref="paper",
            yanchor="top",
            text=f"mediana (pontilhada): {_num(mediana, 1)}<br>média (tracejada): {_num(media, 1)}",
            showarrow=False,
            align="left",
            xanchor="left",
            xshift=8,
            font=dict(color=TINTA, size=12),
        )
    fig.update_layout(barmode="group" if comparando else "relative")
    fig.update_xaxes(title=rotulo_x)
    fig.update_yaxes(title="% dos acidentes", showgrid=True)
    fig = tema(fig, altura=330, legenda=comparando)
    fig.update_layout(bargap=0.12, bargroupgap=0)
    return fig


def boxplot(series: dict[str, pd.Series], rotulo: str) -> go.Figure:
    fig = go.Figure()
    for nome, s in series.items():
        fig.add_box(
            name=nome,
            x=s.dropna(),
            marker=dict(color=COR.get(nome, AZUL), size=4, opacity=0.5),
            line=dict(color=COR.get(nome, AZUL), width=2),
            boxmean=True,
            boxpoints="outliers",
            hoverinfo="x",
        )
    fig.update_xaxes(title=rotulo, showgrid=True)
    fig.update_yaxes(tickfont=dict(color=TINTA))
    return tema(fig, altura=120 + 60 * len(series))


def acidentes_por_ano(series: dict[str, pd.Series]) -> go.Figure:
    comparando = len(series) > 1
    fig = go.Figure()
    for nome, anos in series.items():
        validos = anos.dropna().astype(int)
        f = validos.value_counts(normalize=True).sort_index()
        n = validos.value_counts().sort_index()
        fig.add_bar(
            name=nome,
            x=f.index.astype(str),
            y=f.values,
            marker=dict(color=COR.get(nome, AZUL)),
            customdata=n.values,
            text=[_pct(v, 0 if comparando else 1) for v in f.values],
            textposition="outside",
            textfont=dict(color=TINTA, size=12),
            cliponaxis=False,
            hovertemplate="%{x}<br>" + nome + ": %{y:.1%} (%{customdata} acidentes)<extra></extra>",
        )
    fig.update_yaxes(tickformat=".0%", showgrid=True)
    fig.update_xaxes(tickfont=dict(color=TINTA))
    fig.update_layout(barmode="group")
    return tema(fig, altura=340, legenda=comparando)


def mapa_estados(estados: pd.Series, nome: str) -> go.Figure:
    cont = estados.dropna().value_counts()
    escala = ESCALA_LARANJA if nome == "Amostra" else ESCALA_AZUL
    fig = go.Figure(
        go.Choropleth(
            locations=cont.index,
            z=cont.values,
            locationmode="USA-states",
            colorscale=[[i / (len(escala) - 1), c] for i, c in enumerate(escala)],
            marker_line_color="white",
            marker_line_width=0.6,
            colorbar=dict(title=dict(text="acidentes", font=dict(size=12)), thickness=10, len=0.6, outlinewidth=0),
            hovertemplate="%{location}: %{z} acidentes<extra></extra>",
        )
    )
    fig.update_geos(scope="usa", bgcolor=TRANSPARENTE, showlakes=False, landcolor="#eef0ee", subunitcolor="white")
    fig = tema(fig, altura=340)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig


# ---------------------------------------------------------------- inferência


def curva_teste(r: est.ResultadoTeste) -> go.Figure:
    """Densidade da estatística sob H₀, com região crítica sombreada e estatística observada."""
    if r.distribuicao == "normal":
        dist = stats.norm()
        nome = "Z"
    else:
        dist = stats.t(r.graus_liberdade)
        nome = "T"
    lim = 4.0
    x = np.linspace(-lim, lim, 600)
    y = dist.pdf(x)
    fig = go.Figure()
    fig.add_scatter(x=x, y=y, mode="lines", line=dict(color=TINTA_SUAVE, width=2), name=f"distribuição de {nome} se H₀ for verdadeira", hoverinfo="skip")
    c = r.critico
    regioes = {"menor": [(-lim, -c)], "maior": [(c, lim)], "diferente": [(-lim, -c), (c, lim)]}[r.alternativa]
    for i, (a, b) in enumerate(regioes):
        xx = np.linspace(a, b, 200)
        fig.add_scatter(
            x=np.r_[xx, xx[::-1]],
            y=np.r_[dist.pdf(xx), np.zeros_like(xx)],
            fill="toself",
            fillcolor="rgba(137,135,129,0.38)",
            line=dict(width=0),
            name=f"região crítica (α = {_num(r.alfa, 2)})",
            showlegend=i == 0,
            hoverinfo="skip",
        )
        cauda_esquerda = a == -lim
        borda = b if cauda_esquerda else a
        fig.add_annotation(
            x=borda,
            y=float(dist.pdf(borda)),
            yshift=12,
            xshift=-4 if cauda_esquerda else 4,
            xanchor="right" if cauda_esquerda else "left",
            text=f"valor crítico: {_num(borda, 3)}",
            showarrow=False,
            font=dict(color=TINTA_SUAVE, size=12),
        )
    obs = r.estatistica
    fora = abs(obs) > lim
    x_obs = float(np.clip(obs, -lim + 0.05, lim - 0.05))
    fig.add_vline(x=x_obs, line=dict(color=LARANJA, width=3))
    texto = f"{nome} da amostra = {_num(obs)}" + (" (fora da escala)" if fora else "")
    fig.add_annotation(
        x=x_obs,
        y=max(y) * 1.07,
        text=texto,
        showarrow=False,
        xanchor="right" if x_obs > 0 else "left",
        xshift=-6 if x_obs > 0 else 6,
        font=dict(color=TINTA),
    )
    fig.update_xaxes(range=[-lim, lim], showgrid=False, tickvals=[-4, -2, 0, 2, 4])
    fig.update_yaxes(visible=False, range=[0, max(y) * 1.18])
    return tema(fig, altura=290, legenda=True)


def _tracos_ic(fig: go.Figure, estimativa, inferior, superior, parametro_pop, formato, y_amostra, y_pop, espessura: int, marcador: int) -> None:
    fig.add_scatter(
        x=[inferior, superior],
        y=[y_amostra, y_amostra],
        mode="lines",
        line=dict(color=LARANJA, width=espessura),
        name="IC 95% da amostra",
        hovertemplate="IC 95%: " + formato(inferior) + " a " + formato(superior) + "<extra></extra>",
    )
    fig.add_scatter(
        x=[estimativa],
        y=[y_amostra],
        mode="markers",
        marker=dict(color=LARANJA, size=marcador, line=dict(color="white", width=2)),
        name="estimativa da amostra",
        hovertemplate="estimativa da amostra: " + formato(estimativa) + "<extra></extra>",
    )
    fig.add_scatter(
        x=[parametro_pop],
        y=[y_pop],
        mode="markers",
        marker=dict(color=AZUL, size=marcador, symbol="diamond", line=dict(color="white", width=2)),
        name="valor na população",
        hovertemplate="população: " + formato(parametro_pop) + "<extra></extra>",
    )


def grafico_ic(
    estimativa: float, inferior: float, superior: float, parametro_pop: float, valor_h0: float, formato, formato_eixo: str = "", rotulo_h0: str | None = None
) -> go.Figure:
    """Intervalo de confiança da amostra frente ao valor da população e ao valor de H₀."""
    fig = go.Figure()
    _tracos_ic(fig, estimativa, inferior, superior, parametro_pop, formato, "Amostra", "População", 6, 14)
    fig.add_vline(x=valor_h0, line=dict(color=CINZA, width=2, dash="dash"))
    fig.add_annotation(x=valor_h0, y=1.0, yref="paper", yanchor="bottom", text="H₀ = " + (rotulo_h0 or formato(valor_h0)), showarrow=False, font=dict(color=TINTA))
    fig.add_vline(x=parametro_pop, line=dict(color=AZUL, width=1, dash="dot"))
    fig.update_yaxes(categoryorder="array", categoryarray=["População", "Amostra"], tickfont=dict(color=TINTA), range=[-0.6, 1.6])
    fig.update_xaxes(showgrid=True, tickformat=formato_eixo)
    return tema(fig, altura=200, topo=30)


def ic_compacto(
    estimativa: float, inferior: float, superior: float, parametro_pop: float, valor_h0: float, formato, formato_eixo: str = "", rotulo_h0: str | None = None
) -> go.Figure:
    """Versão pequena do gráfico de IC, para o resumo das quatro hipóteses."""
    fig = go.Figure()
    _tracos_ic(fig, estimativa, inferior, superior, parametro_pop, formato, 1, 0, 6, 12)
    fig.add_vline(x=valor_h0, line=dict(color=CINZA, width=2, dash="dash"))
    fig.add_annotation(x=valor_h0, y=1.0, yref="paper", yanchor="bottom", text="H₀ = " + (rotulo_h0 or formato(valor_h0)), showarrow=False, font=dict(color=TINTA, size=12))
    valores = [inferior, superior, parametro_pop, valor_h0]
    folga = (max(valores) - min(valores)) * 0.14
    fig.update_xaxes(showgrid=True, tickformat=formato_eixo, nticks=5, range=[min(valores) - folga, max(valores) + folga], tickfont=dict(size=11))
    fig.update_yaxes(visible=False, range=[-0.7, 1.7])
    return tema(fig, altura=112, topo=22)


def grafico_simulacao(sim: pd.DataFrame, parametro_pop: float, formato, formato_eixo: str = "") -> go.Figure:
    """Muitos ICs, cada um de uma AAS diferente; os que não cobrem o valor da população ficam em destaque."""
    fig = go.Figure()
    for contem, cor, nome, largura in [(True, CINZA_CLARO, "IC contém o valor da população", 2), (False, LARANJA, "IC não contém", 3)]:
        s = sim[sim["contem"] == contem]
        xs, ys = [], []
        for _, linha in s.iterrows():
            xs += [linha["inferior"], linha["superior"], None]
            ys += [linha["amostra"], linha["amostra"], None]
        fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=cor, width=largura), name=f"{nome} ({len(s)})", hoverinfo="skip")
        fig.add_scatter(
            x=s["estimativa"],
            y=s["amostra"],
            mode="markers",
            marker=dict(color=cor, size=4),
            showlegend=False,
            customdata=np.c_[s["inferior"], s["superior"]],
            hovertemplate="sorteio %{y}<br>estimativa: %{x:.3~f}<br>IC: %{customdata[0]:.3~f} a %{customdata[1]:.3~f}<extra></extra>",
        )
    fig.add_vline(x=parametro_pop, line=dict(color=AZUL, width=2))
    fig.add_annotation(
        x=parametro_pop, y=0, yref="paper", yanchor="top", yshift=-24, text="valor na população: " + formato(parametro_pop), showarrow=False, font=dict(color=TINTA, size=12)
    )
    fig.update_yaxes(title="sorteio simulado", showgrid=False, autorange="reversed")
    fig.update_xaxes(showgrid=True, tickformat=formato_eixo)
    fig = tema(fig, altura=520, legenda=True)
    fig.update_layout(margin=dict(b=52))
    return fig


def histograma_valores_p(sim: pd.DataFrame, alfa: float) -> go.Figure:
    fig = go.Figure()
    fig.add_histogram(
        x=sim["valor_p"],
        xbins=dict(start=0, end=1, size=0.05),
        marker=dict(color=LARANJA, line=dict(width=1, color="white")),
        hovertemplate="valor-p entre %{x}<br>%{y} sorteios<extra></extra>",
        name="sorteios",
    )
    fig.add_vline(x=alfa, line=dict(color=CINZA, width=2, dash="dash"))
    fig.add_annotation(x=alfa, y=1.0, yref="paper", yanchor="bottom", text=f"α = {_num(alfa)}", showarrow=False, xanchor="left", font=dict(color=TINTA))
    fig.update_xaxes(title="valor-p", range=[0, 1])
    fig.update_yaxes(title="nº de sorteios", showgrid=True)
    return tema(fig, altura=280, topo=26)
