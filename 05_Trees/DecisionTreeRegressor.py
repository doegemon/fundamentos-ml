# Databricks notebook source
from sklearn import datasets as dt
from sklearn import model_selection as ms
from sklearn import tree as tr
from sklearn import metrics as mt
import numpy as np
from matplotlib import pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC # Criando Conjunto de Dados e Fazendo o _Split_

# COMMAND ----------

X_raw, y_raw = dt.make_regression( n_samples = 20000, n_features = 4, random_state = 0 )

# COMMAND ----------

X_train, X, y_train, y = ms.train_test_split( X_raw, y_raw, test_size = 0.3, random_state = 0 )

print( X_train.shape[0] )
print( X.shape[0] )

# COMMAND ----------

X_test, X_val, y_test, y_val = ms.train_test_split( X, y, test_size = 0.5, random_state = 0)

print( X_test.shape[0] )
print( X_val.shape[0] )

# COMMAND ----------

# MAGIC %md
# MAGIC # Treinamento do Modelo

# COMMAND ----------

# MAGIC %md
# MAGIC ## Diferentes Valores para _max_depth_

# COMMAND ----------

max_depth_values = np.arange( 2, 20, 1 )
mse_list = list()
rmse_list = list()

for i in max_depth_values:

    model = tr.DecisionTreeRegressor( max_depth = i, random_state = 0 )
    model.fit( X_train, y_train )
    y_pred = model.predict( X_val )

    mse = mt.mean_squared_error( y_val, y_pred, squared = True )
    rmse = mt.mean_squared_error( y_val, y_pred, squared = False )

    mse_list.append( mse )
    rmse_list.append( rmse )

# COMMAND ----------

plt.plot( max_depth_values, rmse_list, marker = 'o' )
plt.xlabel( 'Max Depth' )
plt.ylabel( 'RMSE' );

# COMMAND ----------

final_model = tr.DecisionTreeRegressor( max_depth = 14, random_state = 0 )

X_final = np.concatenate( ( X_train, X_val ) )
y_final = np.concatenate( ( y_train, y_val ) )

final_model.fit( X_final, y_final )

y_pred_test = final_model.predict( X_test )

final_rmse = mt.mean_squared_error( y_test, y_pred_test, squared = False )
print( f'Final RMSE: {final_rmse}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Diferentes Valores para _min_samples_leaf_

# COMMAND ----------

min_samples_leaf_values = np.arange( 50, 500, 25 )
mse_list = list()
rmse_list = list()

for i in min_samples_leaf_values:

    model = tr.DecisionTreeRegressor( min_samples_leaf = i, random_state = 0 )
    model.fit( X_train, y_train )
    y_pred = model.predict( X_val )

    mse = mt.mean_squared_error( y_val, y_pred, squared = True )
    rmse = mt.mean_squared_error( y_val, y_pred, squared = False )

    mse_list.append( mse )
    rmse_list.append( rmse )

# COMMAND ----------

plt.plot( min_samples_leaf_values, rmse_list, marker = 'o' )
plt.xlabel( 'Min Samples Leaf' )
plt.ylabel( 'RMSE' );

# COMMAND ----------

final_model = tr.DecisionTreeRegressor( min_samples_leaf = 50, random_state = 0 )

X_final = np.concatenate( ( X_train, X_val ) )
y_final = np.concatenate( ( y_train, y_val ) )

final_model.fit( X_final, y_final )

y_pred_test = final_model.predict( X_test )

final_rmse = mt.mean_squared_error( y_test, y_pred_test, squared = False )
print( f'Final RMSE: {final_rmse}')
