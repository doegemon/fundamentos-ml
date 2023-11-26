# Databricks notebook source
# MAGIC %md
# MAGIC # Ensaio de Machine Learning - Classificação

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Bibliotecas e _Helper Functions_

# COMMAND ----------

import warnings
import numpy  as np
import pandas as pd
from sklearn.tree            import DecisionTreeClassifier
from sklearn.metrics         import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble        import RandomForestClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import GridSearchCV

# COMMAND ----------

def personal_settings():
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.set_option('display.float_format', lambda x:'%.2f' % x)
    warnings.filterwarnings('ignore')


def model_avaliation(model_name, y_true, y_pred):
    acc = accuracy_score( y_true, y_pred )
    precision = precision_score( y_true, y_pred )
    recall = recall_score( y_true, y_pred )
    f1 = f1_score( y_true, y_pred )
    
    return pd.DataFrame( {'Model': model_name,
                         'Accuracy': acc,
                         'Precison': precision,
                         'Recall': recall,
                         'F1-Score': f1}, index=[0] )

personal_settings()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Carregando Dados

# COMMAND ----------

# MAGIC %md
# MAGIC ### Treinamento

# COMMAND ----------

X_train = pd.read_csv( 'data/classificacao/X_train.csv', low_memory=False )
print( X_train.shape )
display( X_train.sample( 5 ) )

# COMMAND ----------

y_train = pd.read_csv( 'data/classificacao/y_train.csv', low_memory=False )
y_train = y_train.values.ravel()
print( y_train.shape )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Validação

# COMMAND ----------

X_val = pd.read_csv( 'data/classificacao/X_validation.csv', low_memory=False )
print( X_val.shape )

# COMMAND ----------

y_val = pd.read_csv( 'data/classificacao/y_validation.csv', low_memory=False )
y_val = y_val.values.ravel()
print( y_val.shape )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Teste

# COMMAND ----------

X_test = pd.read_csv( 'data/classificacao/X_test.csv', low_memory=False )
print( X_test.shape )

# COMMAND ----------

y_test = pd.read_csv( 'data/classificacao/y_test.csv', low_memory=False )
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
# MAGIC #### Logistic Regression

# COMMAND ----------

lr_model = LogisticRegression( random_state=42 )
lr_model.fit( X_train, y_train )

y_pred_lr = lr_model.predict( X_test )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Decision Tree

# COMMAND ----------

tree_model = DecisionTreeClassifier( random_state=42 )
tree_model.fit( X_train, y_train )

y_pred_tree = tree_model.predict( X_test )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Random Forest

# COMMAND ----------

rf_model = RandomForestClassifier( random_state=42 )
rf_model.fit( X_train, y_train )

y_pred_rf = rf_model.predict( X_test )

# COMMAND ----------

# MAGIC %md
# MAGIC #### KNN

# COMMAND ----------

knn_model = KNeighborsClassifier( )
knn_model.fit( X_train, y_train )

y_pred_knn = knn_model.predict( X_test )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Performance

# COMMAND ----------

lr_metrics = model_avaliation( 'Logistic Regression', y_test, y_pred_lr )
tree_metrics = model_avaliation( 'Decision Tree', y_test, y_pred_tree )
rf_metrics = model_avaliation( 'Random Forest', y_test, y_pred_rf )
knn_metrics = model_avaliation( 'KNN', y_test, y_pred_knn )

metrics = pd.concat([lr_metrics, tree_metrics, rf_metrics, knn_metrics])
display( metrics )

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Buscando os Melhores Parâmetros e Treino + Validação + Teste

# COMMAND ----------

# MAGIC %md
# MAGIC #### Logistic Regression

# COMMAND ----------

lr_model = LogisticRegression( C=1.0, solver='lbfgs', max_iter=100, random_state=42 )

params = {"C": [0.5, 1, 2], 
          "max_iter": [50, 100, 200],
          "solver": ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']}

lr_grid = GridSearchCV(lr_model, params, cv=3, verbose=0, scoring='accuracy')

lr_grid.fit(X_train, y_train)

# COMMAND ----------

y_pred_lr_grid = lr_grid.predict( X_val )
lr_grid_metrics = model_avaliation( 'Logistic Regression - Grid Search', y_val, y_pred_lr_grid )
print( f"Logistic Regression - Best Parameters: {lr_grid.best_params_}" )
display( lr_grid_metrics )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Decision Tree

# COMMAND ----------

tree_model = DecisionTreeClassifier( max_depth=5, min_samples_leaf=50, random_state=42 )

params = {"max_depth": [2, 5, 10, 20, 50], 
          "min_samples_leaf": [25, 50, 100, 200, 500, 800]}

tree_grid = GridSearchCV(tree_model, params, cv=3, verbose=0, scoring='accuracy')

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

rf_model = RandomForestClassifier( n_estimators=300, max_depth=5, min_samples_leaf=50, random_state=42 )

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
# MAGIC #### KNN

# COMMAND ----------

knn_model = KNeighborsClassifier( n_neighbors=5 )

params = {"n_neighbors": [3, 5, 7, 10, 15, 20]}

knn_grid = GridSearchCV(knn_model, params, cv=3, verbose=0, scoring='accuracy')

knn_grid.fit(X_train, y_train)

# COMMAND ----------

y_pred_knn_grid = knn_grid.predict( X_val )
knn_grid_metrics = model_avaliation( 'KNN - Grid Search', y_val, y_pred_knn_grid )
print( f"KNN - Best Parameters: {knn_grid.best_params_}" )
display( knn_grid_metrics )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Performance

# COMMAND ----------

grid_metrics = pd.concat([lr_grid_metrics, tree_grid_metrics, rf_grid_metrics, knn_grid_metrics])
display( grid_metrics )

# COMMAND ----------

join_metrics = pd.concat([metrics, grid_metrics])
display(join_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Performance com os Dados de Teste + Melhores Parâmetros

# COMMAND ----------

# DBTITLE 1,Juntando os Dados de Treino + Validação
X_final = np.concatenate( ( X_train, X_val ))
y_final = np.concatenate ( ( y_train, y_val ))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Logistic Regression

# COMMAND ----------

final_lr = LogisticRegression( C=2, max_iter=100, solver='newton-cg', random_state=42 )
final_lr.fit( X_final, y_final )

y_pred_lr_final = final_lr.predict( X_test )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Decision Tree

# COMMAND ----------

final_tree = DecisionTreeClassifier( max_depth=20, min_samples_leaf=25, random_state=42 )
final_tree.fit( X_final, y_final )

y_pred_tree_final = final_tree.predict( X_test )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Random Forest

# COMMAND ----------

final_rf = RandomForestClassifier( random_state=42 )
final_rf.fit( X_final, y_final )

y_pred_rf_final = final_rf.predict( X_test )

# COMMAND ----------

# MAGIC %md
# MAGIC #### KNN

# COMMAND ----------

final_knn = KNeighborsClassifier( n_neighbors=3 )
final_knn.fit( X_final, y_final )

y_pred_knn_final = final_knn.predict( X_test )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Performance

# COMMAND ----------

final_lr_metrics = model_avaliation( 'Logistic Regression - Final', y_test, y_pred_lr_final )
final_tree_metrics = model_avaliation( 'Decision Tree - Final', y_test, y_pred_tree_final )
final_rf_metrics = model_avaliation( 'Random Forest - Final', y_test, y_pred_rf_final )
final_knn_metrics = model_avaliation( 'KNN - Final', y_test, y_pred_knn_final )

final_metrics = pd.concat([final_lr_metrics, final_tree_metrics, final_rf_metrics, final_knn_metrics])
display( final_metrics.sort_values( by='Accuracy', ascending=False) )
