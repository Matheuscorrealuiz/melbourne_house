import pandas as pd
import numpy as np
import json

# arquivo voltado para acesso e manipulação dos dados

# realiza a carga dos dados do arquivo CSV para um dataframe pandas
def load_data():
    # faz a leitura do conjunto de dados
    dados = pd.read_csv('./data/Melbourne_housing_FULL.csv')
    return dados

