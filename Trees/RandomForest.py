# Databricks notebook source
# MAGIC %pip install scikit-plot

# COMMAND ----------

from sklearn import datasets as ds
from sklearn import model_selection as ms
from sklearn import ensemble as en
from sklearn import metrics as mt
import scikitplot as skplt
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC # Criando Conjunto de Dados e Fazendo o _Split_

# COMMAND ----------

X_raw, y_raw = ds.make_classification( n_samples = 20000, n_classes = 2, n_features = 6, n_redundant = 2, random_state = 0 )

# COMMAND ----------

X_train, X, y_train, y = ms.train_test_split( X_raw, y_raw, test_size = 0.2, random_state = 0 )
print( X_train.shape[0] )

X_val, X_test, y_val, y_test = ms.train_test_split( X, y, test_size = 0.5, random_state = 0 )
print( X_val.shape[0] )
print( X_test.shape[0] )

# COMMAND ----------

# MAGIC %md
# MAGIC # Treinamento do Modelo e Performance com Melhor Valor de Threshold

# COMMAND ----------

model = en.RandomForestClassifier( n_estimators = 100, max_depth = 5, max_features = 'sqrt', n_jobs = -1, random_state = 0 )
model.fit( X_train, y_train )

y_pred_val = model.predict( X_val )
y_prob_val = model.predict_proba( X_val )

# COMMAND ----------

acc_val = mt.accuracy_score( y_val, y_pred_val )

print( f"Accuracy over Validation: {acc_val}" )

# COMMAND ----------

skplt.metrics.plot_roc( y_val, y_prob_val );

# COMMAND ----------

skplt.metrics.plot_ks_statistic( y_val, y_prob_val );

# COMMAND ----------

# Best Threshold
best_th = 0.473

# Making the classifications
y_pred_val_besth = (model.predict_proba(X_val)[:,1]>= best_th).astype(int)

# Accuracy using the best threshold
acc_val_besth = mt.accuracy_score( y_val, y_pred_val_besth )

print( f"Accuracy over Validation with Best Threshold: {acc_val_besth}" )

# COMMAND ----------

# MAGIC %md
# MAGIC # _Feature Importance_

# COMMAND ----------

feature_names = [f"feature {i}" for i in range(X.shape[1])]
importances = model.feature_importances_
forest_importances = pd.Series( importances, index = feature_names )

# COMMAND ----------

fig, ax = plt.subplots()
forest_importances.plot.bar( ax=ax )
ax.set_title( 'Feature Importances' )
ax.set_ylabel( 'Mean Decrease in Impurity' )
fig.tight_layout()
