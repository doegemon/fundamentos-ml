# Databricks notebook source
# MAGIC %md
# MAGIC # Aprendizado Supervisionado - Regressão com *Linear Regression*

# COMMAND ----------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import datasets
from sklearn import pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# COMMAND ----------

def round_value(value): 
    return np.round(value, 2)

def model_avaliation(part, real, predict):
    r2 = r2_score(real, predict)
    mse = mean_squared_error(real, predict)
    rmse = mean_squared_error(real, predict, squared=False)

    metrics = {'R2': round_value(r2), 'MSE': round_value(mse), 'RMSE': round_value(rmse)}

    df_aux = pd.DataFrame.from_dict(metrics, orient='index')
    df_aux.rename(columns={0: part}, inplace=True)

    return df_aux.T

# COMMAND ----------

housing = datasets.fetch_california_housing()

df = pd.DataFrame(housing.data, columns = housing.feature_names)
df['Price'] = housing.target
df['Price'] = df[ 'Price' ] * 1000

df.head()

# COMMAND ----------

df.describe()

# COMMAND ----------

df.hist(bins = 25);

# COMMAND ----------

sns.boxplot(x=df['Price']);

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

lr_model = LinearRegression()

model_pipeline = pipeline.Pipeline([("MinMax Scaler", MinMaxScaler()),
                                    ("Linear Regression" , lr_model)])

model_pipeline.fit(X_train, y_train)

# COMMAND ----------

# Fazendo as Previsões
y_pred = model_pipeline.predict(X_train)
y_pred_test = model_pipeline.predict(X_test)

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

train_performance = model_avaliation('Train', y_train, y_pred)
test_performance = model_avaliation('Test', y_test, y_pred_test)

df_metrics = pd.concat([train_performance, test_performance])
df_metrics.T
