import pandas as pd
import streamlit as st

from analise import estatistica as est
from analise import painel

pop = painel.populacao()
amo = painel.amostra()

st.title("Painel 2: Análise descritiva da amostra")
painel.texto(
    f"A mesma análise do Painel 1, feita só com os **{len(amo)} acidentes da amostra aleatória simples**. "
    "Os números desta página são **estimativas**: mudariam um pouco se o sorteio fosse outro. "
    "Nos gráficos, a amostra aparece em laranja e a população em azul."
)

painel.painel_descritivo(amo, "Amostra", referencia=pop)

# ---------------------------------------------------------------- comparação
st.header("A amostra representa bem a população?")


def linha_pct(indicador, p, a):
    return {
        "Indicador": indicador,
        "População (parâmetro)": painel.pct(p),
        "Amostra (estimativa)": painel.pct(a),
        "Diferença": painel.num((a - p) * 100, 1) + " p.p.",
    }


def linha_mph(indicador, p, a):
    return {
        "Indicador": indicador,
        "População (parâmetro)": painel.num(p, 2) + " mph",
        "Amostra (estimativa)": painel.num(a, 2) + " mph",
        "Diferença": painel.num(a - p, 2) + " mph",
    }


vp, va = pop["velocidade_mph"], amo["velocidade_mph"]
fp, fa = pop["ferido"].dropna().astype(bool), amo["ferido"].dropna().astype(bool)
waymo_p, waymo_a = (pop["empresa_grupo"] == "Waymo").mean(), (amo["empresa_grupo"] == "Waymo").mean()

tabela = pd.DataFrame(
    [
        {
            "Indicador": "Número de acidentes",
            "População (parâmetro)": f"N = {painel.inteiro(len(pop))}",
            "Amostra (estimativa)": f"n = {len(amo)}",
            "Diferença": painel.pct(len(amo) / len(pop)) + " da população",
        },
        linha_pct("Robotáxi parado", pop["parado"].mean(), amo["parado"].mean()),
        linha_pct("Com feridos (gravidade conhecida)", fp.mean(), fa.mean()),
        linha_pct("Informados pela Waymo", waymo_p, waymo_a),
        linha_mph("Velocidade média", vp.mean(), va.mean()),
        linha_mph("Velocidade mediana", vp.median(), va.median()),
        linha_mph("Desvio-padrão da velocidade", vp.std(), va.std()),
        linha_mph("3º quartil da velocidade", vp.quantile(0.75), va.quantile(0.75)),
    ]
)

ep_parado = est.ic_proporcao(int(amo["parado"].sum()), len(amo)).erro_padrao
ep_vel = va.std() / len(va.dropna()) ** 0.5
dif_parado = (amo["parado"].mean() - pop["parado"].mean()) * 100
dif_vel = va.mean() - vp.mean()


def comparar(diferenca: float, erro_padrao: float) -> str:
    return "menor" if abs(diferenca) < erro_padrao else "maior"


fatais_pop = int((pop["gravidade"] == "Fatal").sum())
fatais_amo = int((amo["gravidade"] == "Fatal").sum())

c1, c2 = st.columns([3, 2], gap="large")
with c1:
    st.table(tabela.set_index("Indicador"), border="horizontal")
with c2:
    st.markdown(
        f"""
**Sim: as diferenças são do tamanho esperado pelo acaso.**

- **Proporção de robotáxi parado:** diferença de {painel.num(dif_parado, 1)} p.p., {comparar(dif_parado, ep_parado * 100)}
  que o erro padrão da estimativa ({painel.num(ep_parado * 100, 1)} p.p.).
- **Velocidade média:** diferença de {painel.num(dif_vel, 2)} mph, {comparar(dif_vel, ep_vel)} que o erro padrão
  ({painel.num(ep_vel, 2)} mph).
- **Forma das distribuições:** igual à da população, com velocidade concentrada em 0, Waymo dominante e
  maioria sem feridos.
- **Categorias raras:** podem sumir. A população tem {fatais_pop} acidentes fatais, e a amostra tem {fatais_amo}.
  Com n = {len(amo)}, isso é esperado.
"""
    )

painel.rodape()
