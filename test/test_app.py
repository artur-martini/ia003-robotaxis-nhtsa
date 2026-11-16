from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

RAIZ = Path(__file__).resolve().parent.parent
PAGINAS = ["inicio", "populacao", "amostra", "inferencia"]


def abrir(pagina: str, timeout: int = 60) -> AppTest:
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=timeout)
    at.run()
    if pagina != "inicio":
        at.switch_page(f"paginas/{pagina}.py").run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.mark.parametrize("pagina", PAGINAS)
def test_pagina_roda_sem_erro(pagina):
    at = abrir(pagina)
    assert at.title, "a página precisa de um título"


def test_amostra_sem_sobreposicao_e_com_filtro():
    at = abrir("amostra")
    at.toggle(key="amostra_comparar").set_value(False).run()
    assert not at.exception
    at.multiselect(key="amostra_empresas").set_value(["Waymo"]).run()
    assert not at.exception
    at.slider(key="amostra_anos").set_value((2024, 2025)).run()
    assert not at.exception


def test_populacao_troca_variaveis():
    at = abrir("populacao")
    for chave in ["gravidade", "colidiu_com", "estado", "ferido_txt"]:
        at.selectbox(key="população_quali").set_value(chave).run()
        assert not at.exception


@pytest.mark.parametrize("codigo", ["P1", "P2", "M1", "M2"])
def test_inferencia_simulacao_de_cada_hipotese(codigo):
    at = abrir("inferencia", timeout=120)
    at.button_group(key="sim_hipotese").set_value(codigo).run()
    assert not at.exception, [e.value for e in at.exception]
