# Databricks notebook source
# MAGIC %md
# MAGIC # Aprendizado Supervisionado - Regressão com *Linear Regression*

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# from sklearn.metrics import r

# COMMAND ----------

df = pd.read_csv('../ClassificacaoKNN/train.csv')
df.sample(10)

# COMMAND ----------

# Selecionando as features para treinar o modelo
features = ['idade', 'divida_atual', 'renda_anual',
       'valor_em_investimentos', 'taxa_utilizacao_credito', 'num_emprestimos',
       'num_contas_bancarias', 'num_cartoes_credito', 'dias_atraso_dt_venc',
       'num_pgtos_atrasados', 'num_consultas_credito', 'taxa_juros']

# Variável de interesse
label = ['saldo_atual']

# COMMAND ----------

# Dividindo entre Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(df[features], 
                                                    df[label],
                                                    train_size=0.8,
                                                    random_state=42)
print(X_train.shape)
print(X_test.shape)

# COMMAND ----------

# Preparando o Algoritmo
lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

# COMMAND ----------

# Fazendo as Previsões
y_pred = lr_model.predict(X_train)
y_pred_test = lr_model.predict(X_test)

# COMMAND ----------

# Verificando as Previsões em cima dos dados de Treino
df_result = X_train.copy()
df_result['saldo'] = y_train
df_result['previsao'] = y_pred

df_result[['idade', 'saldo', 'previsao']].sample(5)

# COMMAND ----------

# Verificando as Previsões em cima dos dados de Teste
df_result2 = X_test.copy()
df_result2['saldo'] = y_test
df_result2['previsao'] = y_pred_test

df_result2[['idade', 'saldo', 'previsao']].sample(5)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md 
# MAGIC # Dataset SKLearn

# COMMAND ----------

from sklearn import datasets

housing = datasets.fetch_california_housing()

df = pd.DataFrame(housing.data, columns = housing.feature_names)
df['PRICE'] = housing.target

df.head()

# COMMAND ----------

df.columns

# COMMAND ----------

# Selecionando as features para treinar o modelo
features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']

# Variável de interesse
label = ['PRICE']

# COMMAND ----------

# Dividindo entre Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(df[features], 
                                                    df[label],
                                                    train_size=0.8,
                                                    random_state=42)
print(X_train.shape)
print(X_test.shape)

# COMMAND ----------

# Preparando o Algoritmo
lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

# COMMAND ----------

# Fazendo as Previsões
y_pred = lr_model.predict(X_train)
y_pred_test = lr_model.predict(X_test)

# COMMAND ----------

# Verificando as Previsões em cima dos dados de Treino
df_result = X_train.copy()
df_result['preco'] = y_train
df_result['previsao'] = y_pred

df_result[['preco', 'previsao']].sample(5)

# COMMAND ----------

# Verificando as Previsões em cima dos dados de Teste
df_result2 = X_test.copy()
df_result2['preco'] = y_test
df_result2['previsao'] = y_pred_test

df_result2[['preco', 'previsao']].sample(5)
