import streamlit as st

from analise import painel

pop = painel.populacao()

st.title("Painel 1: Análise descritiva da população")
painel.texto(
    f"Todos os **{painel.inteiro(len(pop))} acidentes** com veículo de direção automatizada informados à NHTSA, "
    "de julho de 2021 a setembro de 2026. Os números desta página são **parâmetros**: descrevem a população "
    "inteira, sem incerteza amostral."
)

painel.painel_descritivo(pop, "População")

with st.expander("Definições usadas neste painel"):
    painel.texto(
        """
- **Robotáxi parado:** o veículo autônomo estava parado no trânsito ou estacionado quando a colisão aconteceu.
- **Com feridos:** acidentes com lesão leve, moderada, grave ou fatal, entre os que têm gravidade conhecida.
- **Velocidade:** velocidade do robotáxi imediatamente antes da colisão, informada pela empresa em milhas por
  hora (1 mph ≈ 1,61 km/h).
- **Filtros:** servem para explorar. Os parâmetros usados no Painel 3 consideram a população inteira.
"""
    )

painel.rodape()
