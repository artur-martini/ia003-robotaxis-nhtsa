import pandas as pd

from analise import dados, painel


def _com_textos(df):
    return painel._com_textos(df)


def test_formatacao_brasileira():
    assert painel.num(1234.5, 1) == "1.234,5"
    assert painel.num(-0.32) == "−0,32"
    assert painel.tex(-0.32) == "-0{,}32"
    assert painel.pct(0.5218) == "52,2%"
    assert painel.inteiro(3070) == "3.070"
    assert painel.valor_p(0.00001) == "< 0,0001"


def test_leitura_rapida_usa_os_numeros_do_recorte():
    pop = _com_textos(dados.carregar_populacao())
    frases = painel.leitura_rapida(pop)
    assert len(frases) == 4
    assert painel.pct(pop["parado"].mean()) in frases[0]
    assert "3 foram fatais" in frases[2]
    assert "pela Waymo" in frases[3] and "na Califórnia" in frases[3]


def test_tabela_de_frequencias_com_duas_series():
    pop = _com_textos(dados.carregar_populacao())
    amo = _com_textos(dados.carregar_amostra())
    t = painel.tabela_frequencias({"População": pop["parado_txt"], "Amostra": amo["parado_txt"]}, ["Sim", "Não"])
    assert list(t.columns) == ["Categoria", "População (n)", "População (%)", "Amostra (n)", "Amostra (%)", "Diferença (p.p.)"]
    assert t["Categoria"].tolist() == ["Sim", "Não"]
    assert t.loc[0, "População (n)"] == painel.inteiro(int(pop["parado"].sum()))


def test_tabela_de_medidas_tem_diferenca_quando_compara():
    pop = dados.carregar_populacao()
    amo = dados.carregar_amostra()
    t = painel.tabela_medidas({"População": pop["velocidade_mph"], "Amostra": amo["velocidade_mph"]}, "mph")
    assert list(t.columns) == ["Medida", "População", "Amostra", "Diferença"]
    media = t.set_index("Medida").loc["média"]
    assert media["População"] == painel.num(pop["velocidade_mph"].mean(), 2) + " mph"
    assert isinstance(t, pd.DataFrame)
