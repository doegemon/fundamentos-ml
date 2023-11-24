# Databricks notebook source
# MAGIC %md
# MAGIC # Regressão Logística ( Classificação )

# COMMAND ----------

from sklearn import metrics         as mt
from sklearn import datasets        as dt
from sklearn import linear_model    as lm
from sklearn import model_selection as ms

# COMMAND ----------

# MAGIC %md
# MAGIC ## Criando o Conjunto de Dados e Fazendo o _Split_

# COMMAND ----------

X, y = dt.make_classification(n_samples=10000, 
                           n_features=4, 
                           n_informative=1, 
                           n_redundant=0, 
                           n_clusters_per_class=1,
                            random_state=42)

# COMMAND ----------

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Treinamento e Performance do Modelo

# COMMAND ----------

model = lm.LogisticRegression()

model.fit(X_train, y_train)

# COMMAND ----------

y_pred = model.predict( X_test )

f1 = mt.f1_score( y_test, y_pred )
print( f"F1-Score: {f1}" )
