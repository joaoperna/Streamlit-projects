import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport, compare

# Configuração da página do Streamlit
st.set_page_config(page_title="Classificação de Veículos", layout="wide")


@st.cache_data
def load_data_and_model():
    """
    Função para carregar os dados, treinar o modelo e retornar o encoder, modelo,
    acurácia, dados dos carros, dados de teste e de treino.

    Retornos:
    - encoder: Encoder para transformar categorias em valores numéricos
    - modelo: Modelo treinado de Naive Bayes categórico
    - acuracia: Acurácia do modelo
    - carros: DataFrame com os dados dos carros
    - X_test: Dados de teste (features)
    - X_train: Dados de treino (features)
    - y_test: Labels de teste
    - y_train: Labels de treino
    """
    carros = pd.read_csv("car.csv", sep=",")
    encoder = OrdinalEncoder()

    # Convertendo as colunas para categorias
    for col in carros.columns.drop("class"):
        carros[col] = carros[col].astype("category")

    # Codificando os dados
    X_encoded = encoder.fit_transform(carros.drop("class", axis=1))
    y = carros["class"].astype("category").cat.codes

    # Dividindo os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.3, random_state=42
    )

    # Treinando o modelo
    modelo = CategoricalNB()
    modelo.fit(X_train, y_train)

    # Fazendo previsões e calculando a acurácia
    y_pred = modelo.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)

    return encoder, modelo, acuracia, carros, X_test, X_train, y_test, y_train


# Carregando os dados e o modelo
encoder, modelo, acuracia, carros, X_test, X_train, y_test, y_train = (
    load_data_and_model()
)

# Título e descrição do aplicativo
st.title("Previsão de Qualidade do Veículo")
st.markdown("***")

# Interface para entrada dos dados
input_features = [
    st.selectbox("Preço:", carros["buying"].unique()),
    st.selectbox("Manutenção:", carros["maint"].unique()),
    st.selectbox("Portas:", carros["doors"].unique()),
    st.selectbox("Capacidade de Passageiros:", carros["persons"].unique()),
    st.selectbox("Porta Malas:", carros["lug_boot"].unique()),
    st.selectbox("Segurança:", carros["safety"].unique()),
]

# Processando a entrada e fazendo a previsão
if st.button("Processar"):
    input_df = pd.DataFrame(
        [input_features], columns=carros.columns.drop("class")
    )
    input_encoded = encoder.transform(input_df)
    predict_encoded = modelo.predict(input_encoded)
    previsao = (
        carros["class"].astype("category").cat.categories[predict_encoded][0]
    )
    st.header(f"Resultado da Previsão: {previsao}")

# Exibindo a acurácia do modelo
st.write(f"Acurácia do modelo: {acuracia:.2f}")

# Link para a documentação do Streamlit
st.markdown(
    "[Para mais referências sobre aplicação do Streamlit](https://docs.streamlit.io/en/stable/api.html#display-text)",
    False,
)

# Preparando os dados completos de treino e teste
train_data = pd.DataFrame(X_train, columns=carros.columns.drop("class"))
train_data["class"] = pd.Categorical.from_codes(
    y_train, carros["class"].astype("category").cat.categories
)

test_data = pd.DataFrame(X_test, columns=carros.columns.drop("class"))
test_data["class"] = pd.Categorical.from_codes(
    y_test, carros["class"].astype("category").cat.categories
)

# Gerando e exibindo relatórios de perfil dos dados de treino e teste
train_report = ProfileReport(train_data, title="Train Data Profile")
test_report = ProfileReport(test_data, title="Test Data Profile")
comparison_report = compare([train_report, test_report])
st_profile_report(comparison_report)
