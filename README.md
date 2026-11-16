# Robotáxis em acidentes: população × amostra

IA003 (Estatística) · Trabalho final · Artur Martini da Rosa

Painéis em Streamlit que comparam a população de acidentes com veículos de direção automatizada (ADS) informados à
NHTSA com uma amostra aleatória simples dela. Os painéis trazem a análise descritiva das duas, intervalos de
confiança e testes de hipóteses para duas proporções e duas médias.

## Base de dados

| Item | Valor |
|---|---|
| Fonte | NHTSA (National Highway Traffic Safety Administration), Standing General Order 2021-01: Incident Reports, ADS |
| Página oficial | https://www.nhtsa.gov/laws-regulations/standing-general-order-crash-reporting |
| Arquivo histórico (2021–2025) | https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/Archive-2021-2025/SGO-2021-01_Incident_Reports_ADS.csv |
| Arquivo atual (2025–2026) | https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/SGO-2021-01_Incident_Reports_ADS.csv |
| Dicionário de dados | `dados/brutos/SGO-2021-01_Data_Element_Definitions.pdf` |
| Baixado em | 15/11/2026 (atualização mensal da NHTSA de 15/10/2026) |
| População | 3.070 acidentes distintos (jul/2021 a set/2026) |
| Amostra | AAS sem reposição, n = 342, semente 20261115 |

## Painéis

| Página | URL (depois de publicado) | Conteúdo |
|---|---|---|
| Base e amostragem | `/` | fonte, contexto, tratamento, população, variáveis, tamanho amostral e sorteio da AAS |
| Painel 1: População | `/populacao` | análise descritiva da população (parâmetros) |
| Painel 2: Amostra | `/amostra` | a mesma análise com a amostra, comparada com a população |
| Painel 3: Inferência | `/inferencia` | resumo das quatro hipóteses, detalhe de cada uma, simulação de outros sorteios e limitações |

Nos três painéis, azul é a população, laranja é a amostra e cinza tracejado é o valor testado em H₀.

## Hipóteses (α = 5%, confiança de 95%)

| | Pergunta | H₀ × H₁ |
|---|---|---|
| P1 | A maioria dos acidentes acontece com o robotáxi parado? | π = 0,50 × π > 0,50 |
| P2 | A parcela de acidentes com feridos é menor que a dos acidentes em geral nos EUA (28%)? | π = 0,28 × π < 0,28 |
| M1 | A velocidade média antes da colisão é menor que o limite urbano de 25 mph? | μ = 25 × μ < 25 |
| M2 | Nos acidentes com feridos, o robotáxi andava mais rápido? (Welch) | μ₁ = μ₂ × μ₁ > μ₂ |

Referências dos valores de H₀: NHTSA, *Summary of Motor Vehicle Traffic Crashes: 2023 Data* (DOT HS 813 762,
Tabela 1), em `referencias/`, e California Vehicle Code §22352.

## Como executar

Com [pixi](https://pixi.sh):

```bash
pixi run app              # abre os painéis em http://localhost:8501
pixi run testes           # roda os testes
pixi run baixar-dados     # baixa de novo os CSVs da NHTSA para dados/brutos/
pixi run preparar-dados   # gera dados/populacao_ads.csv
pixi run sortear-amostra  # calcula n e sorteia dados/amostra_ads.csv
```

Com pip (Python 3.12):

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicação no Streamlit Community Cloud

1. Suba esta pasta para um repositório **público** no GitHub (a pasta `.pixi/` fica de fora pelo `.gitignore`).
2. Em https://share.streamlit.io, crie o app apontando para `app.py` e escolha Python 3.12.
3. Os links de entrega ficam `https://<nome-do-app>.streamlit.app/populacao`, `/amostra` e `/inferencia`.

## Aparência

O tema fica em `.streamlit/config.toml`: claro e fixo (não muda com o tema do navegador), com as fontes Overpass
(títulos) e Public Sans (texto), carregadas do Google Fonts. Sem internet, o navegador usa a fonte sem serifa padrão.

## Estrutura

```
app.py                 navegação entre as páginas
paginas/               uma página por painel
analise/dados.py       leitura, deduplicação e tradução dos CSVs da NHTSA
analise/amostragem.py  tamanho da amostra e sorteio da AAS
analise/estatistica.py descritivas, ICs, testes Z e t, Welch e poder
analise/hipoteses.py   definição e avaliação das quatro hipóteses
analise/graficos.py    gráficos Plotly
analise/painel.py      peças de Streamlit compartilhadas
scripts/               baixar, preparar e sortear
test/                  testes (estatística contra o SciPy, dados, peças dos painéis e páginas)
```
