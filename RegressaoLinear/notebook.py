# Databricks notebook source
# MAGIC %md
# MAGIC # Aprendizado Supervisionado - Regressão com *Linear Regression*

# COMMAND ----------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import datasets

# COMMAND ----------

housing = datasets.fetch_california_housing()

df = pd.DataFrame(housing.data, columns = housing.feature_names)
df['Price'] = housing.target

df.head()

# COMMAND ----------

# Selecionando as features para treinar o modelo
features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
# features = ['HouseAge', 'AveRooms']

# Variável de interesse
label = ['Price']

# COMMAND ----------

# Dividindo entre Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(df[features], 
                                                    df[label],
                                                    train_size=0.8,
                                                    random_state=42)
print(X_train.shape)
print(X_test.shape)

# COMMAND ----------

X_train.describe()

# COMMAND ----------

X_train.isna().sum()

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

# COMMAND ----------

# Performance com R²
r2_train = r2_score(y_train, y_pred)
r2_test = r2_score(y_test, y_pred_test)

print(f"R² Train: {r2_train}")
print(f"R² Test: {r2_test}")
