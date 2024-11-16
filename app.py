import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import util
import data_handler
import pickle

# para rodar esse arquivo
# streamlit run app.py

print("Abriu a pagina")


print("Carregou a pagina")

# Aqui começa a estrutura do App que vai ser executado em produção (nuvem AWS)

# primeiro de tudo, carrega os dados para um dataframe
dados = data_handler.load_data()

# carrega o modelo de predição já treinado e validado
model = pickle.load(open('./models/final_classification_model_melbourne.pkl', 'rb'))

# começa a estrutura da interface do sistema
st.title('Melbourne Housing ML')

data_analyses_on = st.toggle('Exibir análise dos dados')

if data_analyses_on:
    # essa parte é só um exmplo de que é possível realizar diversas visualizações e plotagens com o streamlit
    st.header('Melbourne Housing - Dataframe')

    # exibe todo o dataframe dos dados
    st.dataframe(dados.dropna())

    # plota um gráfico de barras com a contagem dos dados
    st.header('Mapa de imóveis')
    st.map(data=dados.dropna(), latitude="Lattitude", longitude="Longtitude")

# daqui em diante vamos montar a inteface para capturar os dados de input do usuário para realizar a predição
# que vai identificar predizer a renda de uma pessoa
st.header('Preditor de imóveis')

# ler as seguintes informações de input:
# age - int
# education-num - int
# hour-per-week - int

# essas foram as informações utilizadas para treinar o modelo
# assim, todas essas informações também devem ser passadas para o modelo realizar a predição

# define a linha 1 de inputs com 3 colunas
col1, col2, col3, col4, col5 = st.columns(5)

# captura a idade da pessoa, como o step é 1, ele considera a idade como inteira
with col1:
    rooms = st.number_input('Número de cômodos', step=1)

# captura os anos investidos em educação da pessoa
with col2:
    bathroom = st.number_input('Número de banheiros', step=1)

# captura a horas trabalhadas por semana
with col3:
    buildArea = st.number_input('Área construção', step=1)

with col4:
    suburb = st.text_input('Bairro')

with col5:
    bedrooms = st.number_input('Número de quartos', step=1)


submit = st.button('Predizer valor do imóvel')

# data mapping
# essa parte do código realiza o mapeamento dos campos
# o mesmo procedimento foi realizado durante o treinamento do modelo
# assim, isso também deve ser feito aqui para haver compatibilidade nos dados

# armazena todos os dados da pessoa nesse dict
house = {}


# verifica se o botão submit foi pressionado
if submit:
    # seta todos os attrs da pessoa e já realiza o mapeamento dos attrs
    # se houver atributos não numéricos, agora é o momento de realizar o mapeamento
    house = {
        'Rooms': rooms,
        'Bathroom': bathroom,
        'BuildingArea': buildArea,
        'Suburb': suburb,
        'Bedroom2': bedrooms,
    }
    print(house)

    # converte a pessoa para um pandas dataframe
    # isso é feito para igualar ao tipo de dado que foi utilizado para treinar o modelo
    values = pd.DataFrame([house])
    print(values)

    # realiza a predição de income da pessoa com base nos dados inseridos pelo usuário
    results = model.predict(values)
    print(results)

    # o modelo foi treinado para retornar uma lista com <=50k e >50k, onde cada posição da lista indica a renda da pessoa
    # como estamos realizando a predição de somente uma pessoa por vez, o modelo deverá retornar somente um elemento na lista
    if len(results) == 1:
        st.subheader(results[0])
