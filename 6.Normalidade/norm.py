import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st

# Configurações iniciais da página do Streamlit
st.set_page_config(page_title="Teste de Normalidade", layout="wide")
st.title("Teste de Normalidade")

# Barra lateral para upload do arquivo e botão de processamento
with st.sidebar:
    upload_file = st.file_uploader(
        "Escolha o arquivo:", type=["csv"], accept_multiple_files=False
    )
    process_button = st.button("Processar")

# Processa o arquivo após o botão ser pressionado e o arquivo ser carregado
if process_button and upload_file is not None:
    try:
        # Lê o arquivo CSV carregado
        data = pd.read_csv(upload_file, header=0)

        # Verifica se o arquivo está vazio ou se a primeira coluna não tem dados válidos
        if data.empty or data.iloc[:, 0].isnull().all():
            st.error(
                "O arquivo está vazio ou a primeira coluna não tem dados válidos"
            )
        else:
            # Cria duas colunas para os gráficos
            col1, col2 = st.columns(2)

            # Gera o histograma
            with col1:
                fig_hist, ax_hist = plt.subplots()
                ax_hist.hist(
                    data.iloc[:, 0].dropna(),
                    bins="auto",
                    color="blue",
                    alpha=0.7,
                    rwidth=0.85,
                )
                ax_hist.set_title("Histograma")
                st.pyplot(fig_hist)

            # Gera o QQ Plot
            with col2:
                fig_qq, ax_qq = plt.subplots()
                stats.probplot(
                    data.iloc[:, 0].dropna(), dist="norm", plot=ax_qq
                )
                ax_qq.set_title("QQ Plot")
                st.pyplot(fig_qq)

            # Realiza o teste de Shapiro-Wilk para normalidade
            shapiro_test = stats.shapiro(data.iloc[:, 0].dropna())
            st.write(f"Valor de P: {shapiro_test.pvalue:.5f}")

            # Interpreta o resultado do teste de Shapiro-Wilk
            if shapiro_test.pvalue > 0.05:
                st.success(
                    "Não existe evidências suficientes para rejeitar a hipótese de normalidade dos dados"
                )
            else:
                st.warning(
                    "Existem evidências suficientes para rejeitar a hipótese de normalidade dos dados"
                )
    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
