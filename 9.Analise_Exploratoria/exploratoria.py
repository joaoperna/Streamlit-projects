import altair as alt
import pandas as pd
import plotly.express as px
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder

# Configuração da página do Streamlit
st.set_page_config(page_title="Análise Exploratória", layout="wide")
st.title("Despesas de Empenho da Rubrica Diárias do País")


# Função para carregar os dados com cache para otimizar o carregamento
@st.cache_data
def load_data():
    dados = pd.read_csv("dados.csv", sep=";")
    dados["PROPORCAO"] = dados["VALOREMPENHO"] / dados["PIB"]
    return dados


# Carregando os dados
dados = load_data()

# Configurações da barra lateral
with st.sidebar:
    st.header("Configurações")
    top_n = st.number_input(
        "Selecione o número de entradas para exibir",
        min_value=1,
        max_value=len(dados),
        value=10,
    )

# Criação das abas para a navegação
tab1, tab2, tab3 = st.tabs(
    ["Visão Geral", "Análises Detalhadas", "Maiores Valores"]
)

# Conteúdo da aba "Visão Geral"
with tab1:
    st.header("Resumo dos Dados")
    gb = GridOptionsBuilder.from_dataframe(dados.describe().reset_index())
    gridOptions = gb.build()
    AgGrid(dados.describe().reset_index(), gridOptions=gridOptions, height=260)

# Conteúdo da aba "Análises Detalhadas"
with tab2:
    st.header("Distribuição dos Dados")
    col1, col2 = st.columns(2)

    # Coluna 1: Histogramas e boxplots do Valor de Empenho
    with col1:
        hist_valor = (
            alt.Chart(dados)
            .mark_bar()
            .encode(
                x=alt.X("VALOREMPENHO:Q", bin=True, title="Valor de Empenho"),
                y=alt.Y("count()", title="Contagem"),
            )
            .properties(title="Histograma do Valor de Empenho")
        )
        st.altair_chart(hist_valor, use_container_width=True)

        fig2 = px.box(
            dados, x="VALOREMPENHO", title="Boxplot do Valor de Empenho"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Coluna 2: Histogramas e boxplots do PIB
    with col2:
        hist_pib = (
            alt.Chart(dados)
            .mark_bar()
            .encode(
                x=alt.X("PIB:Q", bin=True, title="PIB"),
                y=alt.Y("count()", title="Contagem"),
            )
            .properties(title="Histograma do PIB")
        )
        st.altair_chart(hist_pib, use_container_width=True)
        fig4 = px.box(dados, x="PIB", title="Boxplot do PIB")
        st.plotly_chart(fig4, use_container_width=True)


# Conteúdo da aba "Maiores Valores"
with tab3:
    st.header("Maiores Valores")
    col1, col2, col3 = st.columns(3)

    # Coluna 1: Maiores valores de Empenho
    with col1:
        m_empenho = dados.nlargest(top_n, "VALOREMPENHO")
        bar_empenho = (
            alt.Chart(m_empenho)
            .mark_bar()
            .encode(
                x=alt.X("MUNICIPIO:N", title="Município"),
                y=alt.Y("VALOREMPENHO:Q", title="Valor de Empenho"),
                tooltip=["MUNICIPIO", "VALOREMPENHO"],
            )
            .properties(title="Maiores Empenhos")
        )
        st.altair_chart(bar_empenho, use_container_width=True)

    # Coluna 2: Maiores valores de PIB
    with col2:
        m_pibs = dados.nlargest(top_n, "PIB")
        bar_pibs = (
            alt.Chart(m_pibs)
            .mark_bar()
            .encode(
                x=alt.X("MUNICIPIO:N", title="Município"),
                y=alt.Y("PIB:Q", title="PIB"),
                tooltip=["MUNICIPIO", "PIB"],
            )
            .properties(title="Maiores PIB's")
        )
        st.altair_chart(bar_pibs, use_container_width=True)

    # Coluna 3: Maiores proporções de Empenho em relação ao PIB
    with col3:
        m_prop = dados.nlargest(top_n, "PROPORCAO")
        pie_prop = (
            alt.Chart(m_prop)
            .mark_arc()
            .encode(
                theta=alt.Theta(
                    field="PROPORCAO", type="quantitative", title="Proporção"
                ),
                color=alt.Color(
                    field="MUNICIPIO", type="nominal", title="Município"
                ),
                tooltip=["MUNICIPIO", "PROPORCAO"],
            )
            .properties(title="Maiores por Proporções ao PIB")
        )
        st.altair_chart(pie_prop, use_container_width=True)
