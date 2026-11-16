import pandas as pd
import streamlit as st

from analise import dados
from analise import estatistica as est
from analise import graficos as gr
from analise import hipoteses as hp
from analise import painel
from analise.painel import inteiro, num, pct, tex, valor_p

pop = painel.populacao()
amo = painel.amostra()
N, n = len(pop), len(amo)


@st.cache_data
def avaliacoes():
    return hp.avaliar_todas(amo, pop)


@st.cache_data
def simulacao(codigo: str, repeticoes: int) -> pd.DataFrame:
    return hp.simular_amostras(hp.HIPOTESES[codigo], pop, n, repeticoes)


AV = avaliacoes()


# ---------------------------------------------------------------- formatação


def fmt(h: hp.Hipotese):
    """Formatador de valores da hipótese (proporção em %, média em mph)."""
    if h.tipo == "proporcao":
        return lambda v: pct(v, 1)
    return lambda v: num(v, 2) + " mph"


def eixo(h: hp.Hipotese) -> str:
    return ".0%" if h.tipo == "proporcao" else ""


def nome_estatistica(t: est.ResultadoTeste) -> str:
    return "Z" if t.distribuicao == "normal" else "T"


def decisao_txt(t: est.ResultadoTeste, inicio_maiusculo: bool = True) -> str:
    texto = "Rejeita H₀" if t.rejeita_h0 else "Não rejeita H₀"
    return texto if inicio_maiusculo else texto[0].lower() + texto[1:]


def fmt_h0(h: hp.Hipotese) -> str:
    """Valor testado em H₀, sem casas decimais desnecessárias."""
    if h.tipo == "proporcao":
        return pct(h.valor_h0, 0)
    return num(h.valor_h0, 0) + " mph"


def hipoteses_txt(h: hp.Hipotese) -> str:
    if h.tipo == "duas_medias":
        return "H₀: μ₁ = μ₂, contra H₁: μ₁ > μ₂"
    return f"H₀: {h.parametro} = {fmt_h0(h)}, contra H₁: {h.parametro} {est.SIMBOLO_ALTERNATIVA[h.alternativa]} {fmt_h0(h)}"


def curto(h: hp.Hipotese):
    """Formatador sem a unidade, para listas de valores na mesma unidade."""
    if h.tipo == "proporcao":
        return lambda v: pct(v, 1)
    return lambda v: num(v, 2)


ROTULOS = {
    "P1": "P1: robotáxi parado",
    "P2": "P2: acidentes com feridos",
    "M1": "M1: velocidade média",
    "M2": "M2: velocidade com e sem feridos",
}


def conferencia(r: hp.Avaliacao) -> tuple[str, str, str]:
    """Rótulo, cor e ícone do selo que confere a decisão com a população."""
    rotulo = r.tipo_de_decisao.split(" (")[0]
    if rotulo.startswith("Decisão correta"):
        return rotulo, "green", ":material/check_circle:"
    return rotulo, "yellow", ":material/warning:"


def selo_decisao(t: est.ResultadoTeste) -> None:
    st.badge(decisao_txt(t), color="primary" if t.rejeita_h0 else "gray")


# ---------------------------------------------------------------- abertura

st.title("Painel 3: Inferência estatística e comparação")
painel.texto(
    f"Quatro perguntas sobre os acidentes com robotáxi: duas sobre **proporções** e duas sobre **médias**. "
    f"Os intervalos de confiança e os testes usam **só os {n} acidentes da amostra**. "
    f"Depois, cada resultado é conferido com o valor real, calculado nos {inteiro(N)} acidentes da população."
)
painel.faixa_metricas(
    [
        ("Nível de significância (α)", "5%", "fixado antes de olhar a amostra"),
        ("Nível de confiança", "95%", "dos intervalos de confiança"),
        ("Amostra (n)", str(n), "aleatória simples, sem reposição"),
        ("População (N)", inteiro(N), "todos os acidentes da base"),
    ]
)

# ---------------------------------------------------------------- resumo das quatro hipóteses

st.header("Resultado das quatro hipóteses")
st.markdown(
    f"Em cada linha: <span style='color:{gr.LARANJA}'>━●━</span> intervalo de confiança de 95% e estimativa da amostra, "
    f"<span style='color:{gr.AZUL}'>◆</span> valor real na população e "
    f"<span style='color:{gr.CINZA}'>┆</span> valor testado em H₀.",
    unsafe_allow_html=True,
)
for c, r in AV.items():
    h, t, ic = r.hipotese, r.teste, r.ic
    f = fmt(h)
    with st.container(border=True, key=f"resumo_{c}"):
        esquerda, meio, direita = st.columns([4, 4, 5], gap="medium", vertical_alignment="center")
        with esquerda:
            st.markdown(f"**{c}. {h.pergunta_curta}**")
            st.caption(hipoteses_txt(h))
        with meio:
            painel.grafico(
                gr.ic_compacto(ic.estimativa, ic.inferior, ic.superior, r.parametro_pop, h.valor_h0, f, eixo(h), fmt_h0(h)),
                key=f"resumo_ic_{c}",
            )
        with direita:
            st.markdown(
                f"<span style='color:{gr.LARANJA}'>●</span> Amostra: **{f(ic.estimativa)}** (IC: {curto(h)(ic.inferior)} a {curto(h)(ic.superior)})  \n"
                f"<span style='color:{gr.AZUL}'>◆</span> População: **{f(r.parametro_pop)}**",
                unsafe_allow_html=True,
            )
            rotulo, cor, icone = conferencia(r)
            with st.container(horizontal=True, gap="small"):
                selo_decisao(t)
                st.badge(rotulo, color=cor, icon=icone)
            st.caption(f"valor-p: {valor_p(t.valor_p)}")

rejeitadas = [c for c, r in AV.items() if r.teste.rejeita_h0]
nao_rejeitadas = [c for c, r in AV.items() if not r.teste.rejeita_h0]
cobertos = sum(r.ic_contem_parametro for r in AV.values())
painel.texto(
    f"""
**Como ler o resultado**

- **{cobertos} dos 4 intervalos de confiança contêm o valor real da população.** As estimativas da amostra são confiáveis.
- **{" e ".join(rejeitadas)} rejeitam H₀, e a população confirma.** Os acidentes com robotáxi têm menos feridos que
  a média nacional e acontecem bem abaixo de 25 mph.
- **{" e ".join(nao_rejeitadas)} não rejeitam H₀, mas na população H₀ é falsa por pouco.** São erros tipo II: a
  diferença real existe, mas é pequena demais para uma amostra de {n} detectar. A última seção desta página mostra
  que esse é o resultado mais provável, e não azar no sorteio.
"""
)

with st.expander("Ver o resumo em tabela"):
    linhas = []
    for c, r in AV.items():
        h = r.hipotese
        f = fmt(h)
        linhas.append(
            {
                "Hipótese": c,
                "H₀ contra H₁": hipoteses_txt(h).replace("H₀: ", "").replace(", contra H₁: ", " × "),
                "Estimativa": f(r.ic.estimativa),
                "IC 95%": f"{f(r.ic.inferior)} a {f(r.ic.superior)}",
                "Estatística": f"{nome_estatistica(r.teste)} = {num(r.teste.estatistica, 2)}",
                "Valor-p": valor_p(r.teste.valor_p),
                "Decisão (α = 5%)": decisao_txt(r.teste),
                "População": f(r.parametro_pop),
                "Conferência": conferencia(r)[0],
            }
        )
    st.table(pd.DataFrame(linhas).set_index("Hipótese"), border="horizontal")

# ---------------------------------------------------------------- detalhe de cada hipótese


def condicoes(r: hp.Avaliacao) -> list[str]:
    h, d = r.hipotese, r.detalhes
    ok, alerta = ":material/check:", ":material/warning:"
    itens = [f"{ok} **Independência:** amostra aleatória simples de {n} acidentes, sem reposição."]
    if h.tipo == "proporcao":
        itens.append(
            f"{ok} **Aproximação normal do teste:** n·π₀ = {num(d['n_pi0'], 1)} e n·(1 − π₀) = {num(d['n_1_menos_pi0'], 1)}, os dois ≥ 10."
        )
        itens.append(f"{ok} **Aproximação normal do IC:** {d['x']} sucessos e {d['n'] - d['x']} fracassos, os dois ≥ 10.")
        if h.codigo == "P2":
            itens.append(f"{ok} **Dados usados:** {d['n']} dos {n} acidentes; {n - d['n']} têm gravidade desconhecida.")
    elif h.tipo == "media":
        itens.append(f"{ok} **Teste t:** σ é desconhecido, então uso o desvio-padrão da amostra, com {d['n'] - 1} graus de liberdade.")
        itens.append(
            f"{ok} **Normalidade da média:** a velocidade é muito assimétrica (assimetria de {num(d['assimetria'], 2)}), "
            f"mas n = {d['n']} é grande, e o Teorema Central do Limite garante a aproximação."
        )
        itens.append(f"{ok} **Dados usados:** {d['n']} dos {n} acidentes; {n - d['n']} não informam a velocidade.")
    else:
        itens.append(f"{ok} **Grupos independentes:** {d['n1']} acidentes com feridos e {d['n2']} sem feridos.")
        itens.append(
            f"{ok} **Teste t de Welch:** os grupos têm tamanhos e desvios-padrão diferentes "
            f"(s₁ = {num(d['dp1'], 2)} e s₂ = {num(d['dp2'], 2)}), então não suponho variâncias iguais."
        )
        itens.append(
            f"{alerta} **Normalidade da média:** o grupo com feridos tem só {d['n1']} acidentes, no limite para dados "
            "tão assimétricos. O resultado pede cautela."
        )
    return itens


def interpretacao(r: hp.Avaliacao) -> list[str]:
    h, t, ic, d = r.hipotese, r.teste, r.ic, r.detalhes
    f = fmt(h)
    if h.codigo == "P1":
        poder = est.poder_teste_proporcao(h.valor_h0, r.parametro_pop, n, h.alternativa, t.alfa)
        n_80 = est.n_para_poder_proporcao(h.valor_h0, r.parametro_pop, h.alternativa, t.alfa, 0.8)
        return [
            f"**Na amostra.** {d['x']} de {d['n']} acidentes ({f(ic.estimativa)}) ocorreram com o robotáxi parado. "
            f"O IC de 95% vai de {f(ic.inferior)} a {f(ic.superior)} e inclui 50%. Com valor-p de {valor_p(t.valor_p)}, "
            "**não rejeito H₀**: a amostra não comprova que seja a maioria.",
            f"**Na população.** A proporção real é {f(r.parametro_pop)}. H₀ é falsa, e a decisão foi um **erro tipo II**. "
            f"A diferença real, de {num((r.parametro_pop - h.valor_h0) * 100, 1)} p.p., é menor que a margem de erro de "
            f"5 p.p. usada para dimensionar a amostra. Com n = {n}, o teste só detectaria essa diferença em cerca de "
            f"{pct(poder, 0)} das amostras. Para detectar em 80% delas, seriam precisos cerca de {inteiro(n_80)} "
            "acidentes, mais que a população inteira.",
            "**Conclusão.** Cerca de metade dos acidentes acontece com o robotáxi parado. O intervalo de confiança "
            "mostra isso melhor que o teste, porque contém o valor real.",
        ]
    if h.codigo == "P2":
        return [
            f"**Na amostra.** {d['x']} de {d['n']} acidentes com gravidade conhecida ({f(ic.estimativa)}) tiveram "
            f"feridos. O IC de 95%, de {f(ic.inferior)} a {f(ic.superior)}, fica muito abaixo dos 28% do trânsito em "
            f"geral. Com valor-p {valor_p(t.valor_p)}, **rejeito H₀**.",
            f"**Na população.** A proporção real é {f(r.parametro_pop)}, dentro do IC. A decisão está **correta**, e a "
            f"estimativa errou por {num((ic.estimativa - r.parametro_pop) * 100, 1)} p.p.",
            "**Cuidado.** As empresas informam até colisões leves que a polícia nem registra, e os robotáxis rodam em "
            "áreas urbanas, em baixa velocidade. O teste mostra que os acidentes informados são menos graves. Ele não "
            "mostra que o robotáxi é mais seguro por quilômetro rodado, porque a base não tem a quilometragem.",
        ]
    if h.codigo == "M1":
        return [
            f"**Na amostra.** A velocidade média antes da colisão foi de {f(ic.estimativa)} "
            f"({num(ic.estimativa * dados.MPH_PARA_KMH, 1)} km/h), com IC de 95% de {f(ic.inferior)} a {f(ic.superior)}. "
            f"Com T = {num(t.estatistica, 2)} e valor-p {valor_p(t.valor_p)}, **rejeito H₀**: em média, os acidentes "
            "acontecem bem abaixo de 25 mph.",
            f"**Na população.** A média real é {f(r.parametro_pop)}, dentro do IC. A decisão está **correta**, e a "
            f"estimativa errou por {num(ic.estimativa - r.parametro_pop, 2)} mph, menos que um erro padrão "
            f"({num(t.erro_padrao, 2)} mph).",
            "**Cuidado.** A média resume mal esta variável: a mediana é 0, porque em mais da metade dos acidentes o "
            f"robotáxi estava parado. O teste t continua válido pelo Teorema Central do Limite (n = {d['n']}).",
        ]
    poder = est.poder_teste_t_aproximado(r.parametro_pop, t.erro_padrao, h.alternativa, t.alfa)
    dp_pop = pop["velocidade_mph"].std()
    return [
        f"**Na amostra.** Com feridos, a velocidade média foi de {f(d['media1'])} (n₁ = {d['n1']}). Sem feridos, foi de "
        f"{f(d['media2'])} (n₂ = {d['n2']}). A diferença estimada é de {f(ic.estimativa)}, com IC de 95% de "
        f"{f(ic.inferior)} a {f(ic.superior)}, que inclui o zero. Com valor-p de {valor_p(t.valor_p)}, "
        "**não rejeito H₀**.",
        f"**Na população.** A diferença real é de {f(r.parametro_pop)} ({f(d['media1_pop'])} contra "
        f"{f(d['media2_pop'])}). H₀ é falsa por menos de 1 mph, e a decisão foi um **erro tipo II**. Com só "
        f"{d['n1']} acidentes com feridos na amostra, o teste detectaria essa diferença em cerca de {pct(poder, 0)} "
        "das amostras.",
        f"**Conclusão.** A diferença é irrelevante na prática diante de um desvio-padrão de {num(dp_pop, 0)} mph. "
        "A gravidade depende pouco da velocidade do robotáxi, que muitas vezes está parado quando é atingido.",
    ]


def passo_a_passo(r: hp.Avaliacao) -> None:
    h, t, ic, d = r.hipotese, r.teste, r.ic, r.detalhes
    if h.tipo == "proporcao":
        st.latex(
            rf"\hat p = \frac{{{d['x']}}}{{{d['n']}}} = {tex(ic.estimativa, 4)}"
            rf"\qquad Z = \frac{{\hat p - \pi_0}}{{\sqrt{{\pi_0(1-\pi_0)/n}}}} = \frac{{{tex(ic.estimativa, 4)} - {tex(h.valor_h0, 2)}}}"
            rf"{{\sqrt{{{tex(h.valor_h0, 2)}\times{tex(1 - h.valor_h0, 2)}/{d['n']}}}}} = {tex(t.estatistica, 3)}"
        )
        st.latex(
            rf"IC_{{95\%}} = \hat p \pm 1{{,}}96\sqrt{{\hat p(1-\hat p)/n}} = {tex(ic.estimativa, 4)} \pm {tex(ic.margem, 4)}"
            rf" = [{tex(ic.inferior, 4)};\ {tex(ic.superior, 4)}]"
        )
    elif h.tipo == "media":
        st.latex(
            rf"\bar x = {tex(ic.estimativa, 3)},\ s = {tex(d['desvio_padrao'], 3)},\ n = {d['n']}"
            rf"\qquad T = \frac{{\bar x - \mu_0}}{{s/\sqrt n}} = \frac{{{tex(ic.estimativa, 3)} - {tex(h.valor_h0, 0)}}}"
            rf"{{{tex(d['desvio_padrao'], 3)}/\sqrt{{{d['n']}}}}} = {tex(t.estatistica, 3)}"
        )
        st.latex(
            rf"IC_{{95\%}} = \bar x \pm t_{{{d['n'] - 1};\,0{{,}}025}}\,\frac{{s}}{{\sqrt n}} = {tex(ic.estimativa, 3)} \pm "
            rf"{tex(ic.critico, 3)}\times{tex(ic.erro_padrao, 3)} = [{tex(ic.inferior, 3)};\ {tex(ic.superior, 3)}]"
        )
    else:
        st.latex(
            rf"T = \frac{{\bar x_1 - \bar x_2}}{{\sqrt{{s_1^2/n_1 + s_2^2/n_2}}}} = \frac{{{tex(d['media1'], 3)} - {tex(d['media2'], 3)}}}"
            rf"{{\sqrt{{{tex(d['dp1'], 2)}^2/{d['n1']} + {tex(d['dp2'], 2)}^2/{d['n2']}}}}} = {tex(t.estatistica, 3)}"
        )
        st.latex(
            rf"IC_{{95\%}}(\mu_1-\mu_2) = {tex(ic.estimativa, 3)} \pm {tex(ic.critico, 3)}\times{tex(ic.erro_padrao, 3)}"
            rf" = [{tex(ic.inferior, 3)};\ {tex(ic.superior, 3)}]\quad (gl_{{Welch}} = {tex(t.graus_liberdade, 1)})"
        )
    st.markdown(
        f"Região crítica: **{t.regiao_critica.replace('.', ',').replace('-', '−')}**. Valor-p: {valor_p(t.valor_p)}. "
        f"Decisão: **{decisao_txt(t, False)}** ao nível de {num(t.alfa * 100, 0)}%."
    )


st.header("Cada hipótese em detalhe")
abas = st.tabs([ROTULOS[c] for c in AV])
for aba, (c, r) in zip(abas, AV.items()):
    h, t, ic = r.hipotese, r.teste, r.ic
    f = fmt(h)
    with aba:
        st.subheader(h.pergunta)
        tipo = {"proporcao": "uma proporção", "media": "uma média", "duas_medias": "a diferença entre duas médias"}[h.tipo]
        st.caption(f"Parâmetro: {tipo}. Variável: {h.evento}.")

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("**Hipóteses**")
            st.markdown(f"- **H₀:** {h.h0}\n- **H₁:** {h.h1}")
            st.markdown(f"**Por que este valor em H₀?** {h.justificativa}")
            for ref in h.referencias:
                st.caption(f"Fonte: [{ref.texto}]({ref.url}).")
        with c2:
            st.markdown("**Condições de aplicação**")
            st.markdown("\n".join(f"- {item}" for item in condicoes(r)))

        st.markdown("**Resultado na amostra**")
        painel.faixa_metricas(
            [
                ("Estimativa", f(ic.estimativa)),
                ("Intervalo de confiança de 95%", f"{curto(h)(ic.inferior)} a {f(ic.superior)}"),
                (f"Estatística {nome_estatistica(t)}", num(t.estatistica, 3)),
                ("Valor-p", valor_p(t.valor_p)),
            ],
            key=f"resultado_{c}",
            larguras=[1, 1.7, 1, 1],
        )
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            selo_decisao(t)
            st.caption(
                f"O valor-p ({valor_p(t.valor_p)}) é {'menor' if t.rejeita_h0 else 'maior'} que α = 0,05. "
                f"A estatística {'cai' if t.rejeita_h0 else 'não cai'} na região crítica "
                f"({t.regiao_critica.replace('.', ',').replace('-', '−')})."
            )

        g1, g2 = st.columns(2, gap="large")
        with g1:
            painel.grafico(gr.curva_teste(t), "Onde a estatística da amostra cai, se H₀ for verdadeira", key=f"curva_{c}")
        with g2:
            painel.grafico(
                gr.grafico_ic(ic.estimativa, ic.inferior, ic.superior, r.parametro_pop, h.valor_h0, f, eixo(h), fmt_h0(h)),
                "Intervalo de confiança da amostra e valor real na população",
                key=f"ic_{c}",
            )
            dif = ic.estimativa - r.parametro_pop
            erro = (num(dif * 100, 1) + " p.p.") if h.tipo == "proporcao" else (num(dif, 2) + " mph")
            rotulo, cor, icone = conferencia(r)
            st.markdown(
                f"O valor real na população é **{f(r.parametro_pop)}**. A estimativa errou por **{erro}**, "
                f"e o IC {'contém' if r.ic_contem_parametro else 'não contém'} o valor real."
            )
            st.badge(rotulo, color=cor, icon=icone)

        st.markdown("**Interpretação**")
        for paragrafo in interpretacao(r):
            painel.texto(paragrafo)
        with st.expander("Ver o cálculo passo a passo"):
            passo_a_passo(r)

# ---------------------------------------------------------------- variabilidade amostral

st.header("E se o sorteio tivesse sido outro?")
painel.texto(
    f"Como a população inteira está disponível, dá para repetir o sorteio muitas vezes, sempre com n = {n}, e ver "
    "como as estimativas, os intervalos e as decisões mudam de uma amostra para outra. Esta simulação não faz parte "
    "da inferência: ela mostra, na prática, o que significam o nível de confiança e o poder do teste."
)
c1, c2 = st.columns([5, 2], gap="large")
with c1:
    codigo = st.segmented_control("Hipótese", list(hp.HIPOTESES), default="P1", key="sim_hipotese", format_func=ROTULOS.get) or "P1"
with c2:
    repeticoes = st.select_slider("Número de sorteios simulados", options=[200, 500, 1000], value=1000, key="sim_rep")

sim = simulacao(codigo, repeticoes)
r = AV[codigo]
f = fmt(r.hipotese)
cobertura = sim["contem"].mean()
rejeicao = sim["rejeita"].mean()
h0_vale = r.h0_verdadeira_na_populacao
painel.faixa_metricas(
    [
        ("ICs que contêm o valor real", pct(cobertura, 1), "esperado: cerca de 95%"),
        (
            "Sorteios que rejeitam H₀",
            pct(rejeicao, 1),
            "estima o erro tipo I (esperado: 5%)" if h0_vale else "estima o poder do teste",
        ),
        ("Estimativas vão de", f"{curto(r.hipotese)(sim['estimativa'].min())} a {f(sim['estimativa'].max())}", f"valor real: {f(r.parametro_pop)}"),
    ],
    key="sim_metricas",
)

g1, g2 = st.columns([3, 2], gap="large")
with g1:
    mostrar = sim.head(100)
    painel.grafico(
        gr.grafico_simulacao(mostrar, r.parametro_pop, f, eixo(r.hipotese)),
        f"Intervalos de confiança de 95% dos primeiros {len(mostrar)} sorteios",
        key="sim_ics",
    )
with g2:
    painel.grafico(gr.histograma_valores_p(sim, hp.ALFA), f"Valores-p nos {inteiro(repeticoes)} sorteios", key="sim_valores_p")
    if h0_vale:
        st.markdown("Na população, H₀ é verdadeira. Rejeitá-la seria um erro tipo I, que deve ocorrer em cerca de 5% dos sorteios.")
    else:
        st.markdown(
            f"Na população, H₀ é falsa. Por isso, **{pct(rejeicao, 0)} dos sorteios chegam à decisão correta** "
            f"(o poder do teste), e {pct(1 - rejeicao, 0)} cometem erro tipo II."
        )

leituras = {
    "P1": f"**O erro tipo II da nossa amostra é o resultado mais comum.** A proporção real ({f(r.parametro_pop)}) fica "
    "muito perto de 50%, e a maioria dos sorteios não consegue separar as duas. Os intervalos, ainda assim, contêm o "
    "valor real em cerca de 95% dos sorteios. A cobertura fica um pouco acima de 95% porque a amostra é 11% da "
    "população e as fórmulas não usam a correção para população finita.",
    "P2": "**Quando a diferença é grande, a amostra sobra.** Com a proporção real perto de 10% e H₀ em 28%, "
    "praticamente todos os sorteios rejeitam H₀.",
    "M1": "**O Teorema Central do Limite funciona.** Mesmo com a velocidade tão assimétrica, cerca de 95% dos "
    "intervalos contêm a média real. Todos os sorteios rejeitam H₀, porque 25 mph fica muito longe dela.",
    "M2": "**Poucos acidentes com feridos, muita variação.** Cada sorteio tem só cerca de 30 acidentes com feridos, "
    "e a diferença estimada vai de negativa a positiva. Quase nenhum sorteio detecta a diferença real, de "
    f"{f(r.parametro_pop)}. A cobertura fica um pouco abaixo de 95% porque, com tão poucos casos de uma variável "
    "assimétrica, a aproximação normal ainda não é boa.",
}
painel.destaque(leituras[codigo], largura=painel.LARGURA_TEXTO + 60)

# ---------------------------------------------------------------- limitações

st.header("Limitações")
painel.texto(
    f"""
- **Dados declarados.** As próprias empresas informam os acidentes. A gravidade é a "alegada", e a NHTSA avisa que
  os relatórios podem estar incompletos ou não verificados.
- **Sem quilometragem.** A base não diz quanto cada frota rodou. Por isso, não dá para comparar a taxa de acidentes
  por quilômetro entre empresas ou com motoristas humanos.
- **Concentração.** Cerca de três quartos dos acidentes são da Waymo. Os resultados refletem principalmente uma empresa.
- **Mudança de formulário em 2025.** Unifiquei as categorias dos dois formatos, mas pode haver pequenas diferenças de critério.
- **Comparação com a média nacional (P2).** A estatística nacional conta acidentes registrados pela polícia. A base de
  robotáxis inclui colisões leves que a polícia muitas vezes não registra.
- **Variabilidade amostral.** Com n = {n}, diferenças pequenas como as de P1 e M2 ficam abaixo do que a amostra
  detecta. Não rejeitar H₀ não prova que H₀ é verdadeira.
"""
)

with st.expander("Nota técnica: correção para população finita"):
    fcf = est.fator_correcao_finita(N, n)
    painel.texto(
        f"A amostra tem {pct(n / N)} da população, acima dos 5% em que a correção para população finita costuma ser "
        f"desprezada. Multiplicar o erro padrão por √((N − n)/(N − 1)) = {num(fcf, 3)} deixaria os intervalos cerca de "
        f"{pct(1 - fcf, 0)} mais estreitos. O painel segue as fórmulas da disciplina, sem a correção. Os intervalos "
        "são, portanto, um pouco conservadores, e nenhuma decisão muda com a correção:"
    )
    linhas = []
    for c in ("P1", "P2", "M1"):
        r = AV[c]
        f = fmt(r.hipotese)
        m = r.ic.margem * fcf
        linhas.append(
            {
                "Hipótese": c,
                "IC 95% sem correção": f"{f(r.ic.inferior)} a {f(r.ic.superior)}",
                "IC 95% com correção": f"{f(r.ic.estimativa - m)} a {f(r.ic.estimativa + m)}",
                "Valor na população": f(r.parametro_pop),
            }
        )
    with st.container(width=painel.LARGURA_TEXTO):
        st.table(pd.DataFrame(linhas).set_index("Hipótese"), border="horizontal")

painel.rodape()
