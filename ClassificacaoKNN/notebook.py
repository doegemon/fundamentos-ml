# Databricks notebook source
# MAGIC %md
# MAGIC # Aprendizado Supervisionado - Classificação com *KNN*

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

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
model = KNeighborsClassifier(n_neighbors=k)

model.fit(X_train, y_train)

# COMMAND ----------

# Fazendo as Classificações
y_pred = model.predict(X_train)
y_pred_test = model.predict(X_test)

# COMMAND ----------

# Verificando as Classificações em cima dos dados de Treino
df_result = X_train.copy()
df_result['limite_adicional'] = y_train
df_result['classificacao'] = y_pred

df_result.sample(10)

# COMMAND ----------

# Verificando as Classificações em cima dos dados de Teste
df_result2 = X_test.copy()
df_result2['limite_adicional'] = y_test
df_result2['classificacao'] = y_pred_test

df_result2.sample(10)

# COMMAND ----------

# Matriz de Confusão
matrix_train = confusion_matrix(y_train, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix_train, display_labels=model.classes_)
disp.plot()
plt.show()

# COMMAND ----------

# Matriz de Confusão
matrix_test = confusion_matrix(y_test, y_pred_test)
disp2 = ConfusionMatrixDisplay(confusion_matrix=matrix_test, display_labels=model.classes_)
disp2.plot()
plt.show()

# COMMAND ----------

# Acurácia
acc_train = accuracy_score(y_train, y_pred, normalize=True)
acc_test = accuracy_score(y_test, y_pred_test, normalize = True)

print(acc_train)
print(acc_test)
