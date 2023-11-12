# Databricks notebook source
# MAGIC %md
# MAGIC # _Decision Tree_ e Classificação - Métricas de Avaliação

# COMMAND ----------

from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conjunto de Dados e _Split_

# COMMAND ----------

X, y = make_classification( n_samples = 500, 
                           n_classes = 2,
                           n_features = 5,
                           n_redundant = 2,
                           random_state = 0 )

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2 )

# COMMAND ----------

# MAGIC %md
# MAGIC ## _Decision Tree_

# COMMAND ----------

# Modelo
tree_model = DecisionTreeClassifier( max_depth = 3, 
                                    random_state = 42)
# Treinamento
tree_model.fit( X_train, y_train )
# Previsões
y_pred = tree_model.predict( X_test )
y_pred_prob = tree_model.predict_proba( X_test )

# Probabilidade de pertencer à classe 1
y_pred_prob_1 = y_pred_prob[ :, 1 ]

# COMMAND ----------

# Precision vs. Recall
precision, recall, thresholds = precision_recall_curve( y_test, y_pred_prob_1, pos_label = 1)

# COMMAND ----------

# Curva Precision vs. Recall
plt.plot( recall, precision, marker = '.', label = 'Model' )
plt.xlabel( 'Recall' )
plt.ylabel(' Precision' );

# COMMAND ----------

# Curva Thresholds vs. Precision & Recall
plt.plot( thresholds, precision[:-1], 'b--', label = 'Precision' )
plt.plot( thresholds, recall[:-1], 'g-', label = 'Recall' )
plt.xlabel( 'Thresholds' )
plt.ylabel( 'Precision, Recall' )
plt.vlines( 0.515, 0, 1, color = 'r', linestyle = '--')
plt.hlines( 0.9, 0, 1, color = 'r', linestyle = '--')
plt.legend();

# COMMAND ----------

# MAGIC %md
# MAGIC ## _Logistic Regression_

# COMMAND ----------

# Modelo
lr_model = LogisticRegression( solver = 'lbfgs' )
# Treinamento
lr_model.fit( X_train, y_train )
# Previsões
y_pred = lr_model.predict( X_test )
y_pred_prob = lr_model.predict_proba( X_test )

# Probabilidade de pertencer à classe 1
y_pred_prob_1 = y_pred_prob[ :, 1 ]

# COMMAND ----------

# Precision vs. Recall
precision, recall, thresholds = precision_recall_curve( y_test, y_pred_prob_1, pos_label = 1)

# COMMAND ----------

# Curva Precision vs. Recall
plt.plot( recall, precision, marker = '.', label = 'Model' )
plt.xlabel( 'Recall' )
plt.ylabel(' Precision' );

# COMMAND ----------

# Curva Thresholds vs. Precision & Recall
plt.plot( thresholds, precision[:-1], 'b--', label = 'Precision' )
plt.plot( thresholds, recall[:-1], 'g-', label = 'Recall' )
plt.xlabel( 'Thresholds' )
plt.ylabel( 'Precision, Recall' )
plt.vlines( 0.38, 0, 1, color = 'r', linestyle = '--')
plt.hlines( 0.805, 0, 1, color = 'r', linestyle = '--')
plt.legend();
