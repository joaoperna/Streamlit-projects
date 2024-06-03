import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import poisson

# Configuração da página
st.set_page_config(page_title="Probabilidade de Falha em Equipamentos")
st.title("Probabilidade de Falha em Equipamentos")

# Barra lateral para configurações do usuário
with st.sidebar:
    st.header("Configurações")

    # Seleção do tipo de cálculo
    tipo = st.radio(
        "Selecione o tipo de Cálculo",
        options=["Probabilidade exata", "Menos que", "Mais que"],
    )

    # Entrada do número de ocorrências
    ocorrencia = st.number_input(
        "Ocorrência atual", min_value=1, max_value=99, value=2, step=1
    )

    # Botão para processar os dados
    processar = st.button("Processar")

# Processamento dos dados quando o botão é clicado
if processar:
    lamb = ocorrencia
    inic = lamb - 2
    fim = lamb + 2
    x_vals = np.arange(inic, fim + 1)

    # Determinação da distribuição com base na escolha do usuário
    if tipo == "Probabilidade exata":
        probs = poisson.pmf(x_vals, lamb)
        tit = "Probabilidade de Ocorrência"
    elif tipo == "Menos que":
        probs = poisson.cdf(x_vals, lamb)
        tit = "Probabilidade de Ocorrência Igual ou Menor que"
    else:  # "Mais que"
        probs = poisson.sf(x_vals, lamb)
        tit = "Probabilidade de Ocorrência Maior que"

    # Valores arredondados para exibição
    z_vals = np.round(probs, 4)
    labels = [f"{i} prob.: {p}" for i, p in zip(x_vals, z_vals)]

    # Criação do gráfico de barras com Plotly
    fig = go.Figure(
        data=[
            go.Bar(
                x=x_vals,
                y=probs,
                text=labels,
                textposition="auto",
                marker=dict(color=probs, colorscale="Viridis"),
            )
        ]
    )
    fig.update_layout(
        title=tit,
        xaxis_title="Número de Ocorrências",
        yaxis_title="Probabilidade",
        template="plotly_white",
    )

    # Exibição do gráfico no Streamlit
    st.plotly_chart(fig)
