import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import data_handler
import pickle
from sklearn.preprocessing import LabelEncoder

# para rodar esse arquivo
# streamlit run app.py

print("Abriu a pagina")


print("Carregou a pagina")

# Aqui começa a estrutura do App que vai ser executado em produção (nuvem AWS)

# primeiro de tudo, carrega os dados para um dataframe
dados = data_handler.load_data()

dados = dados.dropna()

# carrega o modelo de predição já treinado e validado
model_suburb = pickle.load(open('./models/final_classification_model_melbourne_suburb.pkl', 'rb'))
model_region = pickle.load(open('./models/final_classification_model_melbourne_region.pkl', 'rb'))


# calculo do suburbio com maior preço medio
most_expensive_id_price = dados.groupby('Suburb')['Price'].mean().idxmax()
most_expensive_price = dados.groupby('Suburb')['Price'].mean().max()

most_expensive_id_area = dados.groupby('Suburb')['BuildingArea'].mean().idxmax()
most_expensive_area = dados.groupby('Suburb')['BuildingArea'].mean().max()

least_expensive_id_area = dados.groupby('Suburb')['BuildingArea'].mean().idxmin()
least_expensive_area = dados.groupby('Suburb')['BuildingArea'].median().min()

# calculo do suburbio com menor preço medio
least_expensive = dados.groupby('Suburb')['Price'].mean().idxmin()
least_expensive_price = dados.groupby('Suburb')['Price'].mean().min()

# calculo do preço medio do suburbio Abbotsford
suburb_abb = 'Abbotsford'
preco_medio_abb = dados[dados['Suburb'] == suburb_abb]['Price'].mean()

# calculo do preço medio do suburbio Airport West
suburb_aw = 'Airport West'
preco_medio_aw = dados[dados['Suburb'] == suburb_aw]['Price'].mean()

#declara label encoder para padronização dos dados
label_encoder1 = LabelEncoder()
label_encoder2 = LabelEncoder()

#padroniza a coluna Suburb para numerico
dados_suburb = dados
dados_region = dados
dados_suburb['Suburb'] = label_encoder1.fit_transform(dados_suburb['Suburb'])
dados_region['Regionname'] = label_encoder2.fit_transform(dados_region['Regionname'])

# começa a estrutura da interface do sistema
st.title('Melbourne Housing ML')

data_analyses_on = st.toggle('Exibir análise dos dados')

if data_analyses_on:
    # essa parte é só um exmplo de que é possível realizar diversas visualizações e plotagens com o streamlit
    st.header('Melbourne Housing - Dataframe')

    # exibe todo o dataframe dos dados
    st.dataframe(dados.dropna())

    st.header('Price')
    st.bar_chart(dados.Price.value_counts())

    # plota um gráfico de barras com a contagem dos dados
    st.header('Year Built')
    st.bar_chart(dados.YearBuilt.value_counts())

    # plota um gráfico de barras com a contagem dos dados
    st.header('Rooms')
    st.bar_chart(dados.Rooms.value_counts())

    # plota um gráfico de barras com a contagem dos dados
    st.header('Suburbio')
    st.bar_chart(dados.Suburb.value_counts())

    st.info(f"### O subúrbio mais barato é **{least_expensive}** \n com preço médio de **${least_expensive_price:,.2f}**", icon="ℹ️")
    st.info(f"### O subúrbio mais caro é **{most_expensive_id_price}** \n com preço médio de **${most_expensive_price:,.2f}**", icon="ℹ️")

    st.info(f"### O subúrbio **{most_expensive_id_area}** tem a maior área construida: \n**{most_expensive_area:,.2f}m²**", icon="ℹ️")
    st.info(f"### O subúrbio **{least_expensive_id_area}** tem a menor área construida: \n**{least_expensive_area:,.2f}m²**", icon="ℹ️")

    st.info(f"### O preço médio no subúrbio {suburb_abb} é \n ${preco_medio_aw:,.2f}", icon="ℹ️")
    st.info(f"### O preço médio no subúrbio {suburb_aw} é \n ${preco_medio_abb:,.2f}", icon="ℹ️")

    # plota um gráfico de barras com a contagem dos dados
    st.header('Regiões')
    st.bar_chart(dados.Regionname.value_counts())

    # plota um gráfico de barras com a contagem dos dados
    st.header('Mapa de imóveis')
    st.map(data=dados.dropna(), latitude="Lattitude", longitude="Longtitude")

# daqui em diante vamos montar a inteface para capturar os dados de input do usuário para realizar a predição
# que vai identificar predizer o valor de um  imovel
st.header('Preditor de imóveis')

# ler as seguintes informações de input:
# essas foram as informações utilizadas para treinar o modelo
# assim, todas essas informações também devem ser passadas para o modelo realizar a predição

# Criando o formulário 1
with st.form(key='formulario1'):
    # Usando colunas para alinhar os campos
    col1, col2 = st.columns(2)  # A coluna 2 será maior
    # Campos do formulário
    with col1:
        rooms = st.number_input("Número de cômodos:", min_value=1, key='rooms_1')
    with col2:
        bathroom = st.number_input("Número de banheiros:", min_value=1, key='bathroom_1')
    with col1:
        build_area = st.number_input("Área construção:", min_value=1, key='build_area_1')
    with col2:
        bedrooms = st.number_input("Número de quartos:", min_value=1, key='bedrooms_1')


    # Validação dos dados
    valid_rooms = rooms >= 1
    valid_bathroom = rooms >= 1
    valid_build_area = rooms >= 1
    valid_bedrooms = rooms >= 1

    # Armazenando a validação no session_state
    st.session_state.form_valid = valid_rooms and valid_bathroom and valid_build_area and valid_bedrooms

    with st.container():
        # define a linha 1 de inputs com 1 colunas
        col1, col2 = st.columns(2)
        with col1:
            suburb = st.selectbox(
                "Suburbio",
                ('Abbotsford', 'Aberfeldie', 'Airport West', 'Albanvale', 'Albert Park',
                'Albion', 'Alphington', 'Altona', 'Altona Meadows', 'Altona North', 'Ardeer',
                'Armadale', 'Ascot Vale', 'Ashburton', 'Ashwood', 'Aspendale',
                'Aspendale Gardens', 'Attwood', 'Avondale Heights', 'Bacchus Marsh',
                'Balaclava', 'Balwyn', 'Balwyn North', 'Bayswater', 'Bayswater North',
                'Beaconsfield', 'Beaconsfield Upper', 'Beaumaris', 'Bellfield' 'Bentleigh',
                'Bentleigh East', 'Berwick', 'Black Rock', 'Blackburn', 'Blackburn North',
                'Blackburn South', 'Bonbeach', 'Boronia', 'Botanic Ridge', 'Box Hill',
                'Braybrook', 'Briar Hill', 'Brighton', 'Brighton East', 'Broadmeadows',
                'Brookfield', 'Brooklyn', 'Brunswick', 'Brunswick East', 'Brunswick West',
                'Bulleen', 'Bullengarook', 'Bundoora', 'Burnley', 'Burnside',
                'Burnside Heights', 'Burwood', 'Burwood East', 'Cairnlea', 'Camberwell',
                'Campbellfield', 'Canterbury', 'Carlton', 'Carlton North', 'Carnegie',
                'Caroline Springs', 'Carrum','Carrum Downs', 'Caulfield', 'Caulfield North',
                'Caulfield South', 'Chadstone', 'Chelsea', 'Chelsea Heights', 'Cheltenham',
                'Chirnside Park', 'Clarinda', 'Clayton', 'Clayton South', 'Clifton Hill',
                'Coburg', 'Coburg North', 'Collingwood', 'Coolaroo', 'Craigieburn',
                'Cranbourne', 'Cranbourne North', 'Cremorne', 'Croydon', 'Croydon Hills',
                'Croydon North', 'Croydon South', 'Dallas', 'Dandenong', 'Dandenong North',
                'Deepdene', 'Deer Park', 'Delahey', 'Derrimut', 'Diamond Creek',
                'Diggers Rest', 'Dingley Village', 'Doncaster', 'Doncaster East', 'Donvale',
                'Doreen', 'Doveton', 'Eaglemont', 'East Melbourne', 'Edithvale', 'Elsternwick',
                'Eltham', 'Eltham North', 'Elwood', 'Emerald', 'Endeavour Hills', 'Epping', 
                'Kurunjang', 'Kooyong', 'Notting Hill')
            )

    # Botão de envio, habilitado apenas se todos os dados forem válidos
    submit_suburb = st.form_submit_button("Predizer valor do imóvel", disabled=not st.session_state.form_valid)

# Criando o formulário 2
with st.form(key='formulario2'):
    # Usando colunas para alinhar os campos
    col1, col2 = st.columns(2)  # A coluna 2 será maior
    # Campos do formulário
    with col1:
        rooms = st.number_input("Número de cômodos:", min_value=1, key='rooms_2')
    with col2:
        bathroom = st.number_input("Número de banheiros:", min_value=1, key='bathroom_2')
    with col1:
        build_area = st.number_input("Área construção:", min_value=1, key='build_area_2')
    with col2:
        bedrooms = st.number_input("Número de quartos:", min_value=1, key='bedrooms_2')


    # Validação dos dados
    valid_rooms = rooms >= 1
    valid_bathroom = rooms >= 1
    valid_build_area = rooms >= 1
    valid_bedrooms = rooms >= 1

    # Armazenando a validação no session_state
    st.session_state.form_valid = valid_rooms and valid_bathroom and valid_build_area and valid_bedrooms

    with st.container():
        # define a linha 1 de inputs com 1 colunas
        col1, col2 = st.columns(2)
        with col1:
            region = st.selectbox(
                "Região",
                ('Eastern Metropolitan', 'Eastern Victoria', 'Northern Metropolitan',
                'Northern Victoria', 'South-Eastern Metropolitan', 'Southern Metropolitan',
                'Western Metropolitan', 'Western Victoria')
            )

    # Botão de envio, habilitado apenas se todos os dados forem válidos
    submit_region = st.form_submit_button("Predizer valor do imóvel", disabled=not st.session_state.form_valid)

# data mapping
# essa parte do código realiza o mapeamento dos campos
# o mesmo procedimento foi realizado durante o treinamento do modelo
# assim, isso também deve ser feito aqui para haver compatibilidade nos dados

# armazena todos os dados da pessoa nesse dict
house = {}

build_area = float(build_area)
bathroom = float(bathroom)
bedrooms = float(bedrooms)
rooms = float(rooms)

# verifica se o botão submit foi pressionado em suburbio
if submit_suburb:
    # se houver atributos não numéricos, realiza o mapeamento
    encoded_suburb = label_encoder1.transform([suburb])[0]
    house = {
        'BuildingArea': build_area,
        'Bathroom': bathroom,
        'Suburb': encoded_suburb,
        'Bedroom2': bedrooms,
        'Rooms': rooms,
    }
    print(house)

    # isso é feito para igualar ao tipo de dado que foi utilizado para treinar o modelo
    values = pd.DataFrame([house])
    print(values)

    # realiza a predição de um valor de imovel com base nos dados inseridos pelo usuário
    results = model_suburb.predict(values)
    print(results)

    if len(results) == 1:
        st.subheader(f"Valor: ${results[0]:,.2f}")

# verifica se o botão submit foi pressionado em regioes
if submit_region:
    # se houver atributos não numéricos, realiza o mapeamento
    encoded_region = label_encoder2.transform([region])[0]
    house = {
        'BuildingArea': build_area,
        'Bathroom': bathroom,
        'Regionname': encoded_region,
        'Bedroom2': bedrooms,
        'Rooms': rooms,
    }
    print(house)

    # isso é feito para igualar ao tipo de dado que foi utilizado para treinar o modelo
    values = pd.DataFrame([house])
    print(values)

    # realiza a predição de um valor de imovel com base nos dados inseridos pelo usuário
    results = model_region.predict(values)
    print(results)

    if len(results) == 1:
        st.subheader(f"Valor: ${results[0]:,.2f}")
