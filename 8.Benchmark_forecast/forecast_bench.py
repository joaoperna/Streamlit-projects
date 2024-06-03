import datetime

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from pmdarima import auto_arima
from statsmodels.tsa.api import ExponentialSmoothing, Holt

# Configuração da página no Streamlit
st.set_page_config(page_title="Benchmark de Séries Temporais", layout="wide")
st.title("Benchmark de Séries Temporais")


def load_data(uploaded_file):
    """Carrega os dados a partir de um arquivo CSV."""
    data = pd.read_csv(uploaded_file, header=None)
    return data


def plot_forecasts(actual, forecasts, titles):
    """Plota as previsões juntamente com os dados reais usando Plotly."""
    actual_length = len(actual)
    forecast_length = len(forecasts[0])

    # Cria a figura do Plotly
    fig = go.Figure()

    # Adiciona os dados reais ao gráfico
    fig.add_trace(
        go.Scatter(
            x=np.arange(actual_length),
            y=actual,
            mode="lines",
            name="Dados Atuais",
        )
    )

    # Adiciona cada previsão ao gráfico
    for forecast, title in zip(forecasts, titles):
        fig.add_trace(
            go.Scatter(
                x=np.arange(actual_length, actual_length + forecast_length),
                y=forecast,
                mode="lines",
                name=title,
            )
        )

    # Configura o layout do gráfico
    fig.update_layout(
        title="Benchmark de Séries Temporais",
        xaxis_title="Período",
        yaxis_title="Valores",
        legend_title="Previsões",
        width=800,
        height=400,
    )

    return fig


def forecast_methods(train, h, methods):
    """Aplica diferentes métodos de previsão e retorna as previsões."""
    forecast = []
    titles = []

    if methods["naive"]:
        naive_forecast = np.tile(train.iloc[-1], h)
        forecast.append(naive_forecast)
        titles.append("Naive")

    if methods["mean"]:
        mean_forecast = np.tile(train.mean(), h)
        forecast.append(mean_forecast)
        titles.append("Mean")

    if methods["drift"]:
        drift_forecast = train.iloc[-1] + (
            np.arange(1, h + 1)
            * ((train.iloc[-1] - train.iloc[0]) / (len(train) - 1))
        )
        forecast.append(drift_forecast)
        titles.append("Drift")

    if methods["holt"]:
        holt_forecast = Holt(train).fit().forecast(h)
        forecast.append(holt_forecast)
        titles.append("Holt")

    if methods["hw"]:
        hw_forecast = (
            ExponentialSmoothing(
                train, seasonal="additive", seasonal_periods=12
            )
            .fit()
            .forecast(h)
        )
        forecast.append(hw_forecast)
        titles.append("HW Additive")

    if methods["arima"]:
        arima_model = auto_arima(
            train, seasonal=True, m=12, suppress_warnings=True
        )
        arima_forecast = arima_model.predict(n_periods=h)
        forecast.append(arima_forecast)
        titles.append("ARIMA")

    return forecast, titles


# Interface do usuário no Streamlit
with st.sidebar:
    uploaded_file = st.file_uploader("Escolha um Arquivo CSV", type="csv")
    if uploaded_file is not None:
        initial_date_range = (
            datetime.date(2000, 1, 1),
            datetime.date(2013, 12, 1),
        )
        data_range = st.date_input(
            "Informe o Período", value=initial_date_range
        )
        forecast_horizon = st.number_input(
            "Informe o Período de Previsão", min_value=1, value=24, step=1
        )

        st.write("Escolha os Métodos de Previsão")
        methods = {
            "naive": st.checkbox("Naive", value=True),
            "mean": st.checkbox("Mean", value=True),
            "drift": st.checkbox("Drift", value=True),
            "holt": st.checkbox("Holt", value=True),
            "hw": st.checkbox("Holt-Winters", value=True),
            "arima": st.checkbox("ARIMA", value=True),
        }
        process_button = st.button("Processar")

if uploaded_file is not None:
    data = load_data(uploaded_file)
    if process_button and len(data_range) == 2:
        col1, col2 = st.columns([1, 4])
        with col1:
            st.dataframe(data)
        with col2:
            with st.spinner("Processando... Por Favor Aguarde!"):
                start_date, end_date = data_range
                train = data.iloc[:, 0]
                forecasts, titles = forecast_methods(
                    train, forecast_horizon, methods
                )
                fig = plot_forecasts(train, forecasts, titles)
                st.plotly_chart(fig)
    elif process_button:
        st.warning("Por favor selecione um período de datas válidos")
else:
    st.sidebar.warning("Faça upload de um arquivo csv")
