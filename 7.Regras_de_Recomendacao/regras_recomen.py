import altair as alt
import pandas as pd
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from st_aggrid import AgGrid, GridOptionsBuilder

# Configurações da página do Streamlit
st.set_page_config(
    page_title="Geração de Regras de Recomendação", layout="wide"
)
st.title("Geração de Regras de Recomendação")

# Barra lateral para upload de arquivo e definição de parâmetros
with st.sidebar:
    uploaded_file = st.file_uploader("Escolha o arquivo", type=["csv"])
    suporte_minimo = st.number_input("Suporte Mínimo", 0.0001, 1.0, 0.01, 0.01)
    confianca_minima = st.number_input(
        "Confiança Mínima", 0.0001, 1.0, 0.2, 0.01
    )
    lift_minimo = st.number_input("Lift Mínimo", 0.0001, 10.0, 1.0, 0.1)
    tamanho_minimo = st.number_input("Tamanho Mínimo", 1, 10, 2, 1)
    processar = st.button("Processar")

# Processamento dos dados e geração das regras de associação
if processar and uploaded_file is not None:
    try:
        # Leitura e processamento das transações
        transactions = [
            line.decode("utf-8").strip().split(",") for line in uploaded_file
        ]
        te = TransactionEncoder()
        te_arry = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_arry, columns=te.columns_)

        # Geração de conjuntos frequentes e regras de associação
        frequent_itemsets = apriori(
            df, min_support=suporte_minimo, use_colnames=True
        )
        regras = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=confianca_minima,
        )
        regras_filtradas = regras[
            (regras["lift"] >= lift_minimo)
            & (regras["antecedents"].apply(lambda x: len(x) >= tamanho_minimo))
        ]

        # Exibição dos resultados
        if not regras_filtradas.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.header("Transações")
                gb = GridOptionsBuilder.from_dataframe(df)
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
                AgGrid(df, gridOptions=gridOptions)

            with col2:
                st.header("Regras Encontradas")
                gb = GridOptionsBuilder.from_dataframe(regras_filtradas)
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
                AgGrid(regras_filtradas, gridOptions=gridOptions)

            st.header("Visualização")
            chart = (
                alt.Chart(regras_filtradas)
                .mark_circle(size=60)
                .encode(
                    x="support",
                    y="confidence",
                    color="lift",
                    tooltip=[
                        "antecedents",
                        "consequents",
                        "support",
                        "confidence",
                        "lift",
                    ],
                )
                .properties(
                    title="Regras de Associação", width=800, height=400
                )
                .interactive()
            )
            st.altair_chart(chart)

            st.header("Resumo das Regras")
            st.write(f"Total de Regras Geradas: {len(regras_filtradas)}")
            st.write(
                f"Suporte Médio: {regras_filtradas['support'].mean():.4f}"
            )
            st.write(
                f"Confiança Média: {regras_filtradas['confidence'].mean():.4f}"
            )
            st.write(f"Lift Médio: {regras_filtradas['lift'].mean():.4f}")

            # Botão para exportar as regras como CSV
            st.download_button(
                label="Exportar Regras como CSV",
                data=regras_filtradas.to_csv(index=False),
                file_name="regras_associacao.csv",
                mime="text/csv",
            )
        else:
            st.write(
                "Nenhuma Regra foi encontrada com os parâmetros definidos"
            )
    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
