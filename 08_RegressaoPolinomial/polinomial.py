# Databricks notebook source
# MAGIC %md
# MAGIC # Regressão Polinomial

# COMMAND ----------

import numpy as np
from sklearn import preprocessing as pp
from sklearn import linear_model  as lm
from matplotlib import pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC ## Criando o Conjunto de Dados

# COMMAND ----------

n = 100
X = np.linspace( -5, 5, num = n )
y = 0.3*X**2 + X + 2 + np.random.normal( size = n )

plt.scatter( X, y );

# COMMAND ----------

# MAGIC %md
# MAGIC ## Treinamento do Modelo e Gráfico da "Reta"

# COMMAND ----------

# Primeiro temos que transformar os dados
poly = pp.PolynomialFeatures( degree = 2 )
X_poly = poly.fit_transform( X.reshape( -1, 1 ) )

# COMMAND ----------

model = lm.LinearRegression()
model.fit( X_poly, y )

y_pred = model.predict( X_poly )

# COMMAND ----------

# Visualizando a 'reta' de regressão
plt.scatter( X, y )
plt.plot( X, y_pred, color = 'red' ); 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualizando Diferentes Graus de Polinômio

# COMMAND ----------

degrees = [1, 2, 15, 30]
coefs = []
intercepts = []
preds = []

for i in degrees:
    poly = pp.PolynomialFeatures( degree = i )
    X_poly = poly.fit_transform( X.reshape( -1, 1 ) )

    model = lm.LinearRegression()
    model.fit( X_poly, y )

    y_pred = model.predict( X_poly )
    preds.append( y_pred )    

# COMMAND ----------

plt.scatter( X, y )
plt.plot(X, preds[0], color = 'red', label = 'Degree 1' )
plt.plot(X, preds[1], color = 'yellow', label = 'Degree 2' )
plt.plot(X, preds[2], color = 'black', label = 'Degree 15' )
plt.plot(X, preds[3], color = 'purple', label = 'Degree 30' )
plt.legend();
