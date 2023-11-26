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
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.metrics       import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline      import Pipeline

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
lr_model_pipeline = Pipeline([('MinMaxScaler', MinMaxScaler()), ('Linear Regression', lr_model)])

lr_model_pipeline.fit( X_train, y_train )

y_pred_lr = lr_model.predict( X_test )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Decision Tree

# COMMAND ----------

tree_model = DecisionTreeRegressor( random_state=42 )
tree_model_pipeline = Pipeline([('MinMaxScaler', MinMaxScaler()), ('Decision Tree', tree_model)])

tree_model_pipeline.fit( X_train, y_train )

y_pred_tree = tree_model_pipeline.predict( X_test )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Random Forest

# COMMAND ----------

rf_model = RandomForestRegressor( random_state=42 )
rf_model_pipeline = Pipeline([('MinMaxScaler', MinMaxScaler()), ('Random Forest', rf_model)])
rf_model_pipeline.fit( X_train, y_train )

y_pred_rf = rf_model_pipeline.predict( X_test )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Polynomial

# COMMAND ----------

poly = PolynomialFeatures()
X_test_poly  = poly.fit_transform( X_test )
X_train_poly = poly.fit_transform( X_train )

# COMMAND ----------

poly_model = LinearRegression()
poly_model.fit( X_train_poly, y_train )

y_pred_poly = poly_model.predict( X_test_poly )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Linear Regression - Lasso

# COMMAND ----------

lr_lasso_model = Lasso()
lr_lasso_model_pipeline = Pipeline([('MinMaxScaler', MinMaxScaler()), ('LR Lasso', lr_lasso_model)])
lr_lasso_model_pipeline.fit( X_train, y_train )

y_pred_lr_lasso = lr_lasso_model_pipeline.predict( X_test )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Linear Regression - Ridge

# COMMAND ----------

lr_ridge_model = Ridge()
lr_ridge_model_pipeline = Pipeline([('MinMaxScaler', MinMaxScaler()), ('LR Ridge', lr_ridge_model)])
lr_ridge_model_pipeline.fit( X_train, y_train )

y_pred_lr_ridge = lr_ridge_model_pipeline.predict( X_test )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Linear Regression - Elastic Net

# COMMAND ----------

lr_elastic_model = ElasticNet()
lr_elastic_model_pipeline = Pipeline([('MinMaxScaler', MinMaxScaler()), ('LR Elastic', lr_elastic_model)])
lr_elastic_model_pipeline.fit( X_train, y_train )

y_pred_lr_elastic = lr_elastic_model_pipeline.predict( X_test )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Polynomial Regression - Lasso

# COMMAND ----------

poly_lasso_model = Lasso()
poly_lasso_model.fit( X_train_poly, y_train )

y_pred_poly_lasso = poly_lasso_model.predict( X_test_poly )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Polynomial Regression - Ridge

# COMMAND ----------

poly_ridge_model = Ridge()
poly_ridge_model.fit( X_train_poly, y_train )

y_pred_poly_ridge = poly_ridge_model.predict( X_test_poly )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Polynomial Regression - Elastic Net

# COMMAND ----------

poly_elastic_model = ElasticNet()
poly_elastic_model.fit( X_train_poly, y_train )

y_pred_poly_elastic = poly_elastic_model.predict( X_test_poly )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Performance

# COMMAND ----------

lr_metrics = model_avaliation( 'Linear Regression', y_test, y_pred_lr )
tree_metrics = model_avaliation( 'Decision Tree', y_test, y_pred_tree )
rf_metrics = model_avaliation( 'Random Forest', y_test, y_pred_rf )
poly_metrics = model_avaliation( 'Polynomial Regression', y_test, y_pred_poly )
lr_lasso_metrics = model_avaliation( 'LR - Lasso', y_test, y_pred_lr_lasso )
lr_ridge_metrics = model_avaliation( 'LR - Ridge', y_test, y_pred_lr_ridge )
lr_elastic_metrics = model_avaliation( 'LR - Elastic', y_test, y_pred_lr_elastic )
poly_lasso_metrics = model_avaliation( 'Polynomial - Lasso', y_test, y_pred_poly_lasso )
poly_ridge_metrics = model_avaliation( 'Polynomial - Ridge', y_test, y_pred_poly_ridge )
poly_elastic_metrics = model_avaliation( 'Polynomial - Elastic', y_test, y_pred_poly_elastic)


metrics = pd.concat([lr_metrics, tree_metrics, rf_metrics, poly_metrics, 
                     lr_lasso_metrics, lr_ridge_metrics, lr_elastic_metrics,
                     poly_lasso_metrics, poly_ridge_metrics, poly_elastic_metrics])
display( metrics.sort_values( by='R²', ascending=False ) )

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Buscando os Melhores Parâmetros e Treino + Validação + Teste

# COMMAND ----------

# MAGIC %md
# MAGIC #### Decision Tree

# COMMAND ----------

tree_model = DecisionTreeRegressor( max_depth=5, min_samples_leaf=50, random_state=42 )

params = {"max_depth": [2, 5, 10, 20, 50], 
          "min_samples_leaf": [25, 50, 100, 200, 500, 800]}

tree_grid = GridSearchCV(tree_model, params, cv=3, verbose=0, scoring='r2')

tree_grid.fit(X_train, y_train)

# COMMAND ----------

y_pred_tree_grid = tree_grid.predict( X_val )
tree_grid_metrics = model_avaliation( 'Decision Tree - Grid Search', y_val, y_pred_tree_grid )
print( f"Decision Tree - Best Parameters: {tree_grid.best_params_}" )
display( tree_grid_metrics )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Random Forest

# COMMAND ----------

rf_model = RandomForestRegressor( n_estimators=300, max_depth=5, min_samples_leaf=50, random_state=42 )

params = {"n_estimators": [50, 100, 200, 300, 500], 
          "min_samples_leaf": [5, 10, 20, 50, 100]}

rf_grid = GridSearchCV(rf_model, params, cv=3, verbose=0, scoring='accuracy')

rf_grid.fit(X_train, y_train)

# COMMAND ----------

y_pred_rf_grid = rf_grid.predict( X_val )
rf_grid_metrics = model_avaliation( 'Random Forest - Grid Search', y_val, y_pred_rf_grid )
print( f"Random Forest - Best Parameters: {rf_grid.best_params_}" )
display( rf_grid_metrics )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Polynomial Regression

# COMMAND ----------

degrees = np.arange(1, 4, 1)
poly_tunning_metrics = pd.DataFrame()

for d in degrees:
    poly = PolynomialFeatures( degree = d )
    X_val_poly   = poly.fit_transform( X_val )
    X_train_poly = poly.fit_transform( X_train )

    model = LinearRegression()
    model.fit( X_train_poly, y_train )

    y_pred = model.predict( X_val_poly ) 

    text = f"Polynomial - Degree {d}"

    metrics = model_avaliation( text, y_val, y_pred )

    poly_tunning_metrics = pd.concat( [poly_tunning_metrics, metrics] )

poly_tunning_metrics = poly_tunning_metrics.sort_values( by='R²', ascending=False )
display( poly_tunning_metrics )
poly_grid_metrics = poly_tunning_metrics.head( 1 )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Linear Regression - Lasso

# COMMAND ----------

lr_lasso_model = Lasso( alpha=1, max_iter=1000 )

params = {"alpha": [0.5, 1, 5, 10, 20, 50], 
          "max_iter": [500, 1000, 2000, 5000]}

lr_lasso_grid = GridSearchCV(lr_lasso_model, params, cv=3, verbose=0, scoring='r2')

lr_lasso_grid.fit(X_train, y_train)

# COMMAND ----------

y_pred_lr_lasso_grid = lr_lasso_grid.predict( X_val )
lr_lasso_grid_metrics = model_avaliation( 'LR Lasso - Grid Search', y_val, y_pred_lr_lasso_grid )
print( f"LR Lasso - Best Parameters: {lr_lasso_grid.best_params_}" )
display( lr_lasso_grid_metrics )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Linear Regression - Ridge

# COMMAND ----------

lr_ridge_model = Ridge( alpha=1, max_iter=1000 )

params = {"alpha": [0.5, 1, 5, 10, 20, 50], 
          "max_iter": [500, 1000, 2000, 5000]}

lr_ridge_grid = GridSearchCV(lr_ridge_model, params, cv=3, verbose=0, scoring='r2')

lr_ridge_grid.fit(X_train, y_train)

# COMMAND ----------

y_pred_lr_ridge_grid = lr_ridge_grid.predict( X_val )
lr_ridge_grid_metrics = model_avaliation( 'LR Ridge - Grid Search', y_val, y_pred_lr_ridge_grid )
print( f"LR Ridge - Best Parameters: {lr_ridge_grid.best_params_}" )
display( lr_ridge_grid_metrics )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Linear Regression - Elastic Net

# COMMAND ----------

lr_elastic_model = ElasticNet( alpha=1, max_iter=1000, l1_ratio=0.5 )

params = {"alpha": [0.5, 1, 5, 10, 20, 50], 
          "max_iter": [500, 1000, 2000, 5000],
          "l1_ratio": [0.25, 0.5, 0.75]}

lr_elastic_grid = GridSearchCV(lr_elastic_model, params, cv=3, verbose=0, scoring='r2')

lr_elastic_grid.fit(X_train, y_train)

# COMMAND ----------

y_pred_lr_elastic_grid = lr_elastic_grid.predict( X_val )
lr_elastic_grid_metrics = model_avaliation( 'LR Elastic - Grid Search', y_val, y_pred_lr_elastic_grid )
print( f"LR Elastic - Best Parameters: {lr_elastic_grid.best_params_}" )
display( lr_elastic_grid_metrics )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Polynomial Regression - Lasso

# COMMAND ----------

degrees = np.arange(1, 4, 1)
poly_lasso_tunning_metrics = pd.DataFrame()

for d in degrees:
    poly = PolynomialFeatures( degree = d )
    X_val_poly   = poly.fit_transform( X_val )
    X_train_poly = poly.fit_transform( X_train )

    model = Lasso( alpha=1, max_iter=1000 )

    params = {"alpha": [0.5, 1, 5, 10, 20, 50], 
          "max_iter": [500, 1000, 2000, 5000]}
    
    model_grid = GridSearchCV(model, params, cv=3, verbose=0, scoring='r2')

    model_grid.fit(X_train_poly, y_train)

    y_pred = model_grid.predict( X_val_poly ) 

    text = f"Polynomial Lasso - Degree {d}"

    metrics = model_avaliation( text, y_val, y_pred )

    print( f"Degree {d} - Best Parameters: {model_grid.best_params_}" )

    poly_lasso_tunning_metrics = pd.concat( [poly_lasso_tunning_metrics, metrics] )

poly_lasso_tunning_metrics = poly_lasso_tunning_metrics.sort_values( by='R²', ascending=False )
display( poly_lasso_tunning_metrics )
poly_lasso_grid_metrics = poly_lasso_tunning_metrics.head( 1 )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Polynomial Regression - Ridge

# COMMAND ----------

degrees = np.arange(1, 4, 1)
poly_ridge_tunning_metrics = pd.DataFrame()

for d in degrees:
    poly = PolynomialFeatures( degree = d )
    X_val_poly   = poly.fit_transform( X_val )
    X_train_poly = poly.fit_transform( X_train )

    model = Ridge( alpha=1, max_iter=1000 )

    params = {"alpha": [0.5, 1, 5, 10, 20, 50], 
          "max_iter": [500, 1000, 2000, 5000]}
    
    model_grid = GridSearchCV(model, params, cv=3, verbose=0, scoring='r2')

    model_grid.fit(X_train_poly, y_train)

    y_pred = model_grid.predict( X_val_poly ) 

    text = f"Polynomial Ridge - Degree {d}"

    metrics = model_avaliation( text, y_val, y_pred )

    print( f"Degree {d} - Best Parameters: {model_grid.best_params_}" )

    poly_ridge_tunning_metrics = pd.concat( [poly_ridge_tunning_metrics, metrics] )

poly_ridge_tunning_metrics = poly_ridge_tunning_metrics.sort_values( by='R²', ascending=False )
display( poly_ridge_tunning_metrics )
poly_ridge_grid_metrics = poly_ridge_tunning_metrics.head( 1 )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Polynomial Regression - Elastic Net

# COMMAND ----------

degrees = np.arange(1, 4, 1)
poly_elastic_tunning_metrics = pd.DataFrame()

for d in degrees:
    poly = PolynomialFeatures( degree = d )
    X_val_poly   = poly.fit_transform( X_val )
    X_train_poly = poly.fit_transform( X_train )

    model = ElasticNet( alpha=1, max_iter=1000, l1_ratio=0.5 )

    params = {"alpha": [0.5, 1, 5, 10, 20, 50], 
          "max_iter": [500, 1000, 2000, 5000],
          "l1_ratio": [0.25, 0.5, 0.75]}
    
    model_grid = GridSearchCV(model, params, cv=3, verbose=0, scoring='r2')

    model_grid.fit(X_train_poly, y_train)

    y_pred = model_grid.predict( X_val_poly ) 

    text = f"Polynomial Elastic - Degree {d}"

    metrics = model_avaliation( text, y_val, y_pred )

    print( f"Degree {d} - Best Parameters: {model_grid.best_params_}" )

    poly_elastic_tunning_metrics = pd.concat( [poly_elastic_tunning_metrics, metrics] )

poly_elastic_tunning_metrics = poly_elastic_tunning_metrics.sort_values( by='R²', ascending=False )
display( poly_elastic_tunning_metrics )
poly_elastic_grid_metrics = poly_elastic_tunning_metrics.head( 1 )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Performance

# COMMAND ----------

grid_metrics = pd.concat([lr_metrics, tree_grid_metrics, rf_grid_metrics, poly_grid_metrics,
                          lr_lasso_grid_metrics, lr_ridge_grid_metrics, lr_elastic_grid_metrics,
                          poly_lasso_grid_metrics, poly_ridge_grid_metrics, poly_elastic_grid_metrics])
display( grid_metrics.sort_values( by='R²', ascending=False ) )

# COMMAND ----------

join_metrics = pd.concat( [metrics, grid_metrics] )
display( join_metrics.sort_values( by='R²', ascending=False ) )

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Performance com os Dados de Teste + Melhores Parâmetros

# COMMAND ----------

# DBTITLE 1,Juntando os Dados de Treino + Validação
#X_final = np.concatenate( ( X_train, X_val ))
#y_final = np.concatenate ( ( y_train, y_val ))
