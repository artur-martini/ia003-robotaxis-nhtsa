import streamlit as st

from analise import painel

st.set_page_config(
    page_title="Acidentes com robotáxis: população × amostra",
    page_icon=":material/directions_car:",
    layout="wide",
)

paginas = st.navigation(
    [
        st.Page("paginas/inicio.py", title="Base e amostragem", icon=":material/dataset:", default=True),
        st.Page("paginas/populacao.py", title="Painel 1: População", icon=":material/groups:", url_path="populacao"),
        st.Page("paginas/amostra.py", title="Painel 2: Amostra", icon=":material/filter_alt:", url_path="amostra"),
        st.Page("paginas/inferencia.py", title="Painel 3: Inferência", icon=":material/functions:", url_path="inferencia"),
    ]
)

with st.sidebar:
    st.html(painel.legenda_cores(len(painel.populacao()), len(painel.amostra())))

paginas.run()
