from datetime import date
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from plotly.subplots import make_subplots
from st_aggrid import AgGrid, GridOptionsBuilder
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Configura a página do Streamlit
st.set_page_config(
    page_title="Sistema de Análise e Previsão de Séries Temporais",
    layout="wide",
)

# Título da aplicação
st.title("Sistema de Análise e Previsão de Séries Temporais")

# Sidebar para upload de arquivo e configuração dos parâmetros
with st.sidebar:
    # Upload do arquivo CSV
    uploaded_file = st.file_uploader("Escolha o arquivo:", type=["csv"])

    if uploaded_file is not None:
        # Lê o arquivo CSV
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        data = pd.read_csv(stringio, header=None)

        # Entrada de data inicial da série
        data_inicio = date(2000, 1, 1)
        periodo = st.date_input("Período Inicial da Série", data_inicio)

        # Entrada para o número de meses de previsão
        periodo_previsao = st.number_input(
            "Informe quantos meses quer prever",
            min_value=1,
            max_value=48,
            value=12,
        )

        # Botão para processar os dados
        processar = st.button("Processar")

# Processamento dos dados e geração das previsões
if uploaded_file is not None and processar:
    try:
        # Criação da série temporal
        ts_data = pd.Series(
            data.iloc[:, 0].values,
            index=pd.date_range(start=periodo, periods=len(data), freq="M"),
        )

        # Decomposição sazonal da série temporal
        decomposicao = seasonal_decompose(ts_data, model="additive")

        # Prepara os dados de decomposição para Plotly
        decomposicao_df = pd.DataFrame(
            {
                "Date": ts_data.index,
                "Observed": decomposicao.observed,
                "Trend": decomposicao.trend,
                "Seasonal": decomposicao.seasonal,
                "Residual": decomposicao.resid,
            }
        ).reset_index(drop=True)

        # Gráficos de decomposição com Plotly
        fig_decomposicao = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=["Observed", "Trend", "Seasonal", "Residual"],
        )

        fig_decomposicao.add_trace(
            go.Scatter(
                x=decomposicao_df["Date"],
                y=decomposicao_df["Observed"],
                name="Observed",
            ),
            row=1,
            col=1,
        )
        fig_decomposicao.add_trace(
            go.Scatter(
                x=decomposicao_df["Date"],
                y=decomposicao_df["Trend"],
                name="Trend",
            ),
            row=2,
            col=1,
        )
        fig_decomposicao.add_trace(
            go.Scatter(
                x=decomposicao_df["Date"],
                y=decomposicao_df["Seasonal"],
                name="Seasonal",
            ),
            row=3,
            col=1,
        )
        fig_decomposicao.add_trace(
            go.Scatter(
                x=decomposicao_df["Date"],
                y=decomposicao_df["Residual"],
                name="Residual",
            ),
            row=4,
            col=1,
        )

        fig_decomposicao.update_layout(height=800, width=800)

        # Gráfico de dispersão com regressão linear para o Resíduo
        x = decomposicao_df["Date"]
        y = decomposicao_df["Residual"].dropna()
        scatter_fig = go.Figure()
        scatter_fig.add_trace(
            go.Scatter(x=x, y=y, mode="markers", name="Residual")
        )

        # Ajuste de uma regressão linear
        z = np.polyfit(np.arange(len(y)), y, 1)
        p = np.poly1d(z)
        scatter_fig.add_trace(
            go.Scatter(
                x=x,
                y=p(np.arange(len(y))),
                mode="lines",
                name="Linear Regression",
            )
        )

        scatter_fig.update_layout(
            title="Residuals with Linear Regression",
            xaxis_title="Date",
            yaxis_title="Residual",
            height=600,
            width=800,
        )

        # Criação e ajuste do modelo SARIMAX
        modelo = SARIMAX(
            ts_data, order=(2, 0, 0), seasonal_order=(0, 1, 1, 12)
        )
        modelo_fit = modelo.fit()

        # Previsão
        previsao = modelo_fit.forecast(steps=periodo_previsao)

        # Prepara os dados de previsão para Plotly
        previsao_df = pd.DataFrame(
            {
                "Date": pd.date_range(
                    start=ts_data.index[-1],
                    periods=periodo_previsao + 1,
                    freq="M",
                )[1:],
                "Forecast": previsao,
            }
        )

        ts_data_df = ts_data.reset_index().rename(
            columns={"index": "Date", 0: "Value"}
        )

        # Gráfico de previsão com Plotly
        fig_previsao = go.Figure()
        fig_previsao.add_trace(
            go.Scatter(
                x=ts_data_df["Date"],
                y=ts_data_df["Value"],
                mode="lines",
                name="Observado",
            )
        )
        fig_previsao.add_trace(
            go.Scatter(
                x=previsao_df["Date"],
                y=previsao_df["Forecast"],
                mode="lines",
                name="Previsão",
                line=dict(dash="dash", color="red"),
            )
        )

        fig_previsao.update_layout(
            title="Previsão",
            xaxis_title="Data",
            yaxis_title="Valor",
            height=600,
            width=800,
        )

        # Divisão da página em três colunas
        col1, col2, col3 = st.columns([4, 4, 4])

        # Exibição dos resultados
        with col1:
            st.markdown(
                "<h2 style='text-align: center;'>Decomposição</h2>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig_decomposicao, use_container_width=True)
            st.markdown(
                "<h2 style='text-align: center;'>Resíduos com Regressão Linear</h2>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(scatter_fig, use_container_width=True)

        with col2:
            st.markdown(
                "<h2 style='text-align: center;'>Previsão</h2>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig_previsao, use_container_width=True)

        with col3:
            st.markdown("## Dados da Previsão")
            gb = GridOptionsBuilder.from_dataframe(previsao_df)
            gb.configure_pagination()
            gb.configure_side_bar()
            gb.configure_default_column(
                groupable=True,
                value=True,
                enableRowGroup=True,
                aggFunc="sum",
                editable=True,
            )
            gridOptions = gb.build()
            AgGrid(
                previsao_df,
                gridOptions=gridOptions,
                enable_enterprise_modules=True,
            )

    except Exception as e:
        st.error(f"Erro ao processar os dados: {e}")
