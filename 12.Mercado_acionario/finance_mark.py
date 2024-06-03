from datetime import date

import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Configuração da página do Streamlit
st.set_page_config(page_title="Visualizador de Ações", layout="wide")
st.title("Visualizador de Ações")

# Barra lateral para seleção de parâmetros
with st.sidebar:
    empresa_selecionada = st.selectbox(
        "Selecione a empresa para visualizar:",
        [
            "AAPL",
            "GOOGL",
            "MSFT",
            "AMZN",
            "TSLA",
            "PETR4.SA",
            "VALE3.SA",
            "ITUB4.SA",
            "BBDC4.SA",
        ],
    )
    start_date = st.date_input("Data de Início", value=date(2020, 1, 1))
    end_date = st.date_input("Data de Fim", value=date(2020, 1, 15))
    gerar_graficos = st.button("Gerar Gráficos")


# Função para gerar gráficos
def gerar_graficos_acoes(empresa, inicio, fim):
    data = yf.download(empresa, start=inicio, end=fim)
    if data.empty:
        st.error("Erro ao carregar dados, por favor tente novamente")
        return

    # Criação das abas para visualização dos dados
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Preço Fechado Ajustado", "Volume", "Gráfico de Velas", "Dados"]
    )

    # Gráfico de Preço Fechado Ajustado
    with tab1:
        fig_close = go.Figure()
        fig_close.add_trace(
            go.Scatter(
                x=data.index,
                y=data["Adj Close"],
                mode="lines",
                name="Preço Fechado Ajustado",
            )
        )
        fig_close.update_layout(
            title=f"Histórico de Preços para {empresa}",
            xaxis_title="Data",
            yaxis_title="Preço",
        )
        st.plotly_chart(fig_close, use_container_width=True)

    # Gráfico de Volume
    with tab2:
        fig_volume = go.Figure()
        fig_volume.add_trace(go.Bar(x=data.index, y=data["Volume"]))
        fig_volume.update_layout(
            title=f"Volume de Negociação para a empresa {empresa}",
            xaxis_title="Data",
            yaxis_title="Volume",
        )
        st.plotly_chart(fig_volume, use_container_width=True)

    # Gráfico de Velas
    with tab3:
        fig_candle = go.Figure(
            data=[
                go.Candlestick(
                    x=data.index,
                    open=data["Open"],
                    high=data["High"],
                    low=data["Low"],
                    close=data["Close"],
                )
            ]
        )
        fig_candle.update_layout(
            title=f"Gráfico de Velas para a empresa {empresa}",
            xaxis_title="Data",
            yaxis_title="Preço",
        )
        st.plotly_chart(fig_candle, use_container_width=True)

    # Exibição dos dados em formato de tabela
    with tab4:
        st.dataframe(data)


# Verifica se o botão para gerar gráficos foi pressionado
if gerar_graficos:
    gerar_graficos_acoes(empresa_selecionada, start_date, end_date)
