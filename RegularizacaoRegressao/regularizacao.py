# Databricks notebook source
# MAGIC %md
# MAGIC # Regularização em Regressões

# COMMAND ----------

import numpy  as np
import pandas as pd
from sklearn    import datasets     as ds
from sklearn    import linear_model as lm
from matplotlib import pyplot       as plt

# COMMAND ----------

# MAGIC %md
# MAGIC ## Criando o Conjunto de Dados

# COMMAND ----------

n_samples = 100
n_outliers = 5
n_features = 1

X, y, coef = ds.make_regression( n_samples = n_samples, 
                                 n_features = n_features, 
                                 n_informative = 1, 
                                 noise = 10, 
                                 coef = True, 
                                 random_state = 0 )

# Adicionando Outliers
X[:n_outliers] = 3 + 0.5 * np.random.normal( size = ( n_outliers, n_features ) )
y[:n_outliers] = -3 + 10 * np.random.normal( size = ( n_outliers ) )

plt.scatter( X, y );

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modelos - Diferentes Regularizações

# COMMAND ----------

# Linear Regression
lr = lm.LinearRegression()
lr.fit( X, y )

# Lasso - L1
lasso = lm.Lasso( alpha = 20 )
lasso.fit( X, y )

# Ridge - L2
ridge = lm.Ridge( alpha = 20 )
ridge.fit( X, y )

# Elastic Net
elastic = lm.ElasticNet( alpha = 20, l1_ratio = 0.3 )
elastic.fit( X, y )

# RANSAC - algoritmo com identificador de outliers integrado
ransac = lm.RANSACRegressor()
ransac.fit( X, y )

inlier = ransac.inlier_mask_
outlier = np.logical_not( ransac.inlier_mask_ )

# COMMAND ----------

# Conjunto de dados de 'teste' e predições
X_plot = np.arange( X.min(), X.max() )[:, np.newaxis]

y_pred_lr = lr.predict( X_plot )
y_pred_lasso = lasso.predict( X_plot )
y_pred_ridge = ridge.predict( X_plot )
y_pred_elastic = elastic.predict( X_plot )
y_pred_ransac = ransac.predict( X_plot )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gráficos - Visualizando as Retas

# COMMAND ----------

plt.scatter( X, y )
plt.scatter( X[inlier], y[inlier], color='orange' )
plt.scatter( X[outlier], y[outlier], color='blue' )

plt.plot( X_plot, y_pred_lr, color = 'red', label = 'Linear Regression' )
plt.plot( X_plot, y_pred_lasso, color = 'purple', label = 'Lasso' )
plt.plot( X_plot, y_pred_ridge, color = 'green', label = 'Ridge' )
plt.plot( X_plot, y_pred_elastic, color = 'yellow', label = 'Elastic Net' )
plt.plot( X_plot, y_pred_ransac, color = 'black', label = 'RANSAC' )
plt.legend();

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualizando os Coeficientes

# COMMAND ----------

print( f"Coeficiente Original: {coef}" )
print( f"Coeficiente RANSAC: {ransac.estimator_.coef_}" )
print( f"Coeficiente Linear Regression: {lr.coef_}" )
print( f"Coeficiente Ridge: {ridge.coef_}" )
print( f"Coeficiente Lasso: {lasso.coef_}" )
print( f"Coeficiente Elastic Net: {elastic.coef_}" )
