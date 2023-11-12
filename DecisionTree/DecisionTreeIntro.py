# Databricks notebook source
# MAGIC %md
# MAGIC # _Decision Tree_ - Primeiros Passos

# COMMAND ----------

# MAGIC %pip install graphviz opencv-python

# COMMAND ----------

import cv2
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from matplotlib import pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conjunto de Dados

# COMMAND ----------

iris = load_iris()

X = iris.data[ :, 2: ]
y = iris.target

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sem Limite de Crescimento

# COMMAND ----------

model_tree = DecisionTreeClassifier()

model_tree.fit( X, y)

# COMMAND ----------

export_graphviz( model_tree, 
                out_file = 'tree.dot', 
                feature_names = iris.feature_names[ 2: ],
                class_names = iris.target_names,
                rounded = True,
                filled = True)

# COMMAND ----------

!dot -Tpng tree.dot -o tree.png

# COMMAND ----------

# Visualização da árvore
img = cv2.imread( 'tree.png' )
plt.figure( figsize = ( 20, 20 ) )
plt.imshow( img )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Com Limite de Crescimento

# COMMAND ----------

model_tree2 = DecisionTreeClassifier(max_depth = 5,
                                     min_samples_leaf = 30,
                                     max_features = 2,
                                     random_state = 42)

model_tree2.fit( X, y)

# COMMAND ----------

export_graphviz( model_tree2, 
                out_file = 'tree2.dot', 
                feature_names = iris.feature_names[ 2: ],
                class_names = iris.target_names,
                rounded = True,
                filled = True)

# COMMAND ----------

!dot -Tpng tree2.dot -o tree2.png

# COMMAND ----------

# Visualização da árvore
img2 = cv2.imread( 'tree2.png' )
plt.figure( figsize = ( 20, 20 ) )
plt.imshow( img2 )
