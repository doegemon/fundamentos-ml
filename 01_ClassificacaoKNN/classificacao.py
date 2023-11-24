# Databricks notebook source
# MAGIC %md
# MAGIC # Aprendizado Supervisionado - Classificação com *KNN*

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

# COMMAND ----------

df = pd.read_csv('train.csv')
df.sample(10)

# COMMAND ----------

display(df['limite_adicional'].value_counts())
display(df['limite_adicional'].value_counts(normalize=True))

# COMMAND ----------

# Selecionando as features para treinar o modelo
features = ['idade', 'saldo_atual', 'divida_atual', 'renda_anual',
       'valor_em_investimentos', 'taxa_utilizacao_credito', 'num_emprestimos',
       'num_contas_bancarias', 'num_cartoes_credito', 'dias_atraso_dt_venc',
       'num_pgtos_atrasados', 'num_consultas_credito', 'taxa_juros']

# Label para classificação
label = ['limite_adicional']

# COMMAND ----------

# Dividindo entre Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(df[features], 
                                                    df[label],
                                                    train_size=0.8,
                                                    random_state=42)
print(X_train.shape)
print(X_test.shape)

# COMMAND ----------

display(y_train.value_counts(normalize=True))
display(y_test.value_counts(normalize=True))

# COMMAND ----------

# Preparando o Algoritmo
k = 3
knn_model = KNeighborsClassifier(n_neighbors=k)

model_pipeline = Pipeline([("MinMax Scaler", MinMaxScaler()),
                                    ("KNN" , knn_model)])

model_pipeline.fit(X_train, y_train)

# COMMAND ----------

# Fazendo as Classificações
y_pred = model_pipeline.predict(X_train)
y_pred_test = model_pipeline.predict(X_test)

# COMMAND ----------

# Verificando as Classificações em cima dos dados de Treino
df_result = X_train.copy()
df_result['limite_adicional'] = y_train
df_result['classificacao'] = y_pred

df_result[['limite_adicional', 'classificacao']].sample(10)

# COMMAND ----------

# Verificando as Classificações em cima dos dados de Teste
df_result2 = X_test.copy()
df_result2['limite_adicional'] = y_test
df_result2['classificacao'] = y_pred_test

df_result2[['limite_adicional', 'classificacao']].sample(10)

# COMMAND ----------

# Matriz de Confusão
matrix_train = confusion_matrix(y_train, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix_train, display_labels=model_pipeline.classes_)
disp.plot()
plt.show()

# COMMAND ----------

# Matriz de Confusão
matrix_test = confusion_matrix(y_test, y_pred_test)
disp2 = ConfusionMatrixDisplay(confusion_matrix=matrix_test, display_labels=model_pipeline.classes_)
disp2.plot()
plt.show()

# COMMAND ----------

def round_value(value): 
    return np.round(value, 2)

def model_avaliation(part, real, predict, positive_label):
    acc = accuracy_score(real, predict)
    precision = precision_score(real, predict, pos_label = positive_label)
    recall = recall_score(real, predict, pos_label = positive_label)

    metrics = {'Accuracy': round_value(acc), 'Precision': round_value(precision), 'Recall': round_value(recall)}

    df_aux = pd.DataFrame.from_dict(metrics, orient='index')
    df_aux.rename(columns={0: part}, inplace=True)

    return df_aux.T

# COMMAND ----------

train_performance = model_avaliation('Train', y_train, y_pred, 'Conceder')
test_performance = model_avaliation('Test', y_test, y_pred_test, 'Conceder')

df_metrics = pd.concat([train_performance, test_performance])
df_metrics.T

# COMMAND ----------

# MAGIC %md
# MAGIC ## Testando diferentes valores para *k*

# COMMAND ----------

k_list = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]

df_k = pd.DataFrame()
accuracy_list = []
precision_list = []
recall_list = []

# COMMAND ----------

for n in k_list:
  k = n
  knn_model = KNeighborsClassifier(n_neighbors=k)

  model_pipeline = Pipeline([("MinMax Scaler", MinMaxScaler()),
                                    ("KNN" , knn_model)])

  model_pipeline.fit(X_train, y_train)

  y_pred = model_pipeline.predict(X_train)

  accuracy = accuracy_score(y_train, y_pred)
  precision = precision_score(y_train, y_pred, pos_label='Conceder')
  recall = recall_score(y_train, y_pred, pos_label='Conceder')

  accuracy_list.append(accuracy)
  precision_list.append(precision)
  recall_list.append(recall)

  df = pd.DataFrame({'K': k, 
                     'Accuracy': accuracy, 
                     'Precision': precision,
                     'Recall': recall}, index=[0])
  
  df_k = pd.concat([df_k, df])

display(df_k)
