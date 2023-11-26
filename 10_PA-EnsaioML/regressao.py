# Databricks notebook source
# MAGIC %md
# MAGIC # Ensaio de Machine Learning - Regressão

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Bibliotecas e _Helper Functions_

# COMMAND ----------

import warnings
import numpy  as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet 
from sklearn.tree         import DecisionTreeRegressor
from sklearn.ensemble     import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics       import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

# COMMAND ----------

def personal_settings():
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.set_option('display.float_format', lambda x:'%.2f' % x)
    warnings.filterwarnings('ignore')


def model_avaliation(model_name, y_true, y_pred):
    r2 = r2_score( y_true, y_pred )
    mse = mean_squared_error( y_true, y_pred )
    rmse = mean_squared_error( y_true, y_pred, squared=False )
    mae = mean_absolute_error( y_true, y_pred )
    mape = mean_absolute_percentage_error( y_true, y_pred )
    
    return pd.DataFrame( {'Model': model_name,
                         'R²': r2,
                         'MSE': mse,
                         'RMSE': rmse,
                         'MAE': mae,
                         'MAPE': mape}, index=[0] )

personal_settings()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Carregando Dados

# COMMAND ----------

# MAGIC %md
# MAGIC ### Treinamento

# COMMAND ----------

X_train = pd.read_csv( 'data/regressao/X_train.csv', low_memory=False )
print( X_train.shape )
display( X_train.sample( 5 ) )

# COMMAND ----------

y_train = pd.read_csv( 'data/regressao/y_train.csv', low_memory=False )
y_train = y_train.values.ravel()
print( y_train.shape )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Validação

# COMMAND ----------

X_val = pd.read_csv( 'data/regressao/X_validation.csv', low_memory=False )
print( X_val.shape )

# COMMAND ----------

y_val = pd.read_csv( 'data/regressao/y_validation.csv', low_memory=False )
y_val = y_val.values.ravel()
print( y_val.shape )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Teste

# COMMAND ----------

X_test = pd.read_csv( 'data/regressao/X_test.csv', low_memory=False )
print( X_test.shape )

# COMMAND ----------

y_test = pd.read_csv( 'data/regressao/y_test.csv', low_memory=False )
y_test = y_test.values.ravel()
print( y_test.shape )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Treinamento dos Algoritmos

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Sem Alteração nos Parâmetros e Treino + Teste

# COMMAND ----------

# MAGIC %md
# MAGIC #### Linear Regression

# COMMAND ----------

lr_model = LinearRegression()
lr_model.fit( X_train, y_train )

y_pred_lr = lr_model.predict( X_test )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Decision Tree

# COMMAND ----------

tree_model = DecisionTreeRegressor( random_state=42 )
tree_model.fit( X_train, y_train )

y_pred_tree = tree_model.predict( X_test )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Random Forest

# COMMAND ----------

rf_model = RandomForestRegressor( random_state=42 )
rf_model.fit( X_train, y_train )

y_pred_rf = rf_model.predict( X_test )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Polynomial

# COMMAND ----------

poly = PolynomialFeatures()
X_train_poly = poly.fit_transform( X_train )
X_test_poly  = poly.fit_transform( X_test )

# COMMAND ----------

poly_model = LinearRegression()
poly_model.fit( X_train_poly, y_train )

y_pred_poly = poly_model.predict( X_test_poly )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Performance

# COMMAND ----------

lr_metrics = model_avaliation( 'Linear Model', y_test, y_pred_lr )
tree_metrics = model_avaliation( 'Decision Tree', y_test, y_pred_tree )
rf_metrics = model_avaliation( 'Random Forest', y_test, y_pred_rf )
poly_metrics = model_avaliation( 'Polynomial Regression', y_test, y_pred_poly )

metrics = pd.concat([lr_metrics, tree_metrics, rf_metrics, poly_metrics])
display( metrics )
