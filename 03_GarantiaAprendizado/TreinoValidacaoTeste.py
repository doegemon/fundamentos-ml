# Databricks notebook source
# MAGIC %md
# MAGIC # Estratégia de Treino, Validação e Teste em Classificação

# COMMAND ----------

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# COMMAND ----------

# MAGIC %md
# MAGIC ## Criando o Conjunto de Dados

# COMMAND ----------

# DBTITLE 0,Criando o Conjunto de Dados
n_samples = 20000
n_features = 4
n_informative = 4
n_redundant = 0
random_state = 42

X, y = make_classification ( n_samples = n_samples,
                            n_features = n_features,
                            n_informative = n_informative,
                            n_redundant = n_redundant, 
                            random_state = random_state)

df = pd.DataFrame( X )

print( df.shape )
display( df.head() )

# COMMAND ----------

# DBTITLE 1,Separando dados para simular o ambiente de Produção
X_data, X_prod, y_data, y_prod = train_test_split ( X, y, train_size = 0.8, random_state= 42 )

print( X_data.shape[0] )
print( X_prod.shape[0] )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sem Dividir os Dados

# COMMAND ----------

model = DecisionTreeClassifier( max_depth = 38 )

model.fit( X_data, y_data )

y_pred = model.predict( X_data )

acc = accuracy_score( y_data, y_pred )
print(f"Acurácia sobre os dados de Treino: {acc}")

# COMMAND ----------

y_pred_prod = model.predict ( X_prod )

acc_prod = accuracy_score ( y_prod, y_pred_prod )
print(f"Acurácia sobre os dados de Produção: {acc_prod}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Estratégia Treino e Teste

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split ( X_data, y_data, train_size = 0.8, random_state= 42 )

print( X_train.shape[0] )
print( X_test.shape[0] )

# COMMAND ----------

model = DecisionTreeClassifier( max_depth = 38 )

model.fit( X_train, y_train )

y_pred_test = model.predict( X_test )

acc_test = accuracy_score( y_test, y_pred_test )
print(f"Acurácia sobre os dados de Teste: {acc_test}")

# COMMAND ----------

# Escolhendo os melhores parâmetros usando os dados de Teste
values = [i for i in range ( 1, 60 )]
test_score = list()

for i in values: 
    model = DecisionTreeClassifier ( max_depth = i )
    model.fit( X_train, y_train )

    y_pred_test = model.predict( X_test )
    acc_test = accuracy_score( y_test, y_pred_test )

    test_score.append( acc_test )

plt.plot( values, test_score, '-o', label = 'Test' );

# COMMAND ----------

# Modelo Final com o melhor valor para o parâmetro
final_model = DecisionTreeClassifier( max_depth = 14 )

X_final = np.concatenate( ( X_train, X_test ))
y_final = np.concatenate( ( y_train, y_test ))

# Treinamento usando os dados de Treino + Teste
final_model.fit( X_final, y_final )

# COMMAND ----------

# Performance sobre os dados de Produção
y_pred_prod = final_model.predict ( X_prod )

acc_prod = accuracy_score( y_prod, y_pred_prod )
print(f"Acurácia sobre os dados de Produção: {acc_prod}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Estratégia Treino, Validação e Teste

# COMMAND ----------

# Separação do conjunto de Treino entre Treino e Validação
X_train, X_validation, y_train, y_validation = train_test_split ( X_train, y_train, train_size = 0.8, random_state= 42 )

print( X_train.shape[0] )
print( X_validation.shape[0] )

# COMMAND ----------

# Escolhendo os melhores parâmetros
values = [i for i in range ( 1, 60 )]
val_score = list()

for i in values: 
    model = DecisionTreeClassifier ( max_depth = i )
    # Treinando com os dados de Treino
    model.fit( X_train, y_train )

    # Verificando a performance com os dados de Validação
    y_pred_val = model.predict( X_validation )
    acc_val = accuracy_score( y_validation, y_pred_val )

    val_score.append( acc_val )

plt.plot( values, test_score, '-o', label = 'Test' );

# COMMAND ----------

# Modelo Final com o melhor valor para o parâmetro
final_model = DecisionTreeClassifier( max_depth = 13 )

X_final = np.concatenate( ( X_train, X_validation ))
y_final = np.concatenate ( ( y_train, y_validation ))

# Treinamento usando os dados de Treino + Validação
final_model.fit( X_final, y_final )

# COMMAND ----------

# Performance sobre os dados de Teste
y_pred_test= final_model.predict ( X_test )

acc_test = accuracy_score( y_test, y_pred_test )
print(f"Acurácia sobre os dados de Teste: {acc_test}")

# COMMAND ----------

# Modelo final com o melhor valor para o parâmetro
final_model2 = DecisionTreeClassifier( max_depth = 13 )

X_final2 = np.concatenate( ( X_final, X_test ))
y_final2 = np.concatenate ( ( y_final, y_test ))

# Treinamento usando os dados de Treino + Validação + Teste
final_model2.fit( X_final2, y_final2 )

# COMMAND ----------

# Performance sobre os dados de Produção
y_pred_prod = final_model2.predict ( X_prod )

acc_prod = accuracy_score( y_prod, y_pred_prod )
print(f"Acurácia sobre os dados de Produção: {acc_prod}")
