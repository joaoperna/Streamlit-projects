import altair as alt
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# Título da aplicação
st.title(
    "&#8194;" * 6
    + "Previsão Inicial de Custo para "
    + "&#8194;" * 15
    + "Franquia"
)

# Carregamento dos dados
dados = pd.read_csv("slr12.csv", sep=";")

# Definindo as variáveis independentes (X) e dependentes (y)
X = dados[["FrqAnual"]]
y = dados["CusInic"]

# Criando o modelo de regressão linear e ajustando-o aos dados
modelo = LinearRegression().fit(X, y)

# Dividindo a interface em duas colunas
col1, col2 = st.columns([1, 2])

# Exibindo os primeiros 10 registros dos dados na primeira coluna
with col1:
    st.header("&#8194;" * 4 + "Dados")
    st.table(dados.head(10))

# Criando o gráfico de dispersão com a linha de regressão na segunda coluna
with col2:
    st.header("&#8194;" * 6 + "Gráfico de Dispersão")
    base = alt.Chart(dados).encode(
        x=alt.X(
            "FrqAnual",
            scale=alt.Scale(domain=[680, 1400]),
            title="Frequência Anual",
        ),
        y=alt.Y(
            "CusInic",
            scale=alt.Scale(domain=[1100, 1900]),
            title="Custo Inicial",
        ),
    )

    scatter_plot = (
        base.mark_circle(size=60)
        .encode(tooltip=["FrqAnual", "CusInic"])
        .interactive()
    )

    line_plot = base.transform_regression("FrqAnual", "CusInic").mark_line(
        color="red"
    )

    chart = scatter_plot + line_plot
    st.altair_chart(chart, use_container_width=True)

# Entrada de valor anual da franquia pelo usuário
st.header("&#8194;" * 8 + "Valor Anual da Franquia:")
novo_valor = st.number_input(
    "Insira Novo Valor",
    min_value=1.0,
    max_value=999999.0,
    value=1500.0,
    step=0.01,
)
processar = st.button("Processar")

# Processamento e previsão do custo inicial com o novo valor
if processar:
    dados_novo_valor = pd.DataFrame([[novo_valor]], columns=["FrqAnual"])
    prev = modelo.predict(dados_novo_valor)
    st.header(f"Previsão de Custo Inicial R$: {prev[0]:.2f}")
