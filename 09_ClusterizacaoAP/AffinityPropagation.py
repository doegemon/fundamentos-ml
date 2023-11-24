# Databricks notebook source
# MAGIC %md
# MAGIC # Clusterização com Affinity Propagation

# COMMAND ----------

import time
import numpy as np
from sklearn import cluster as c
from sklearn import datasets as ds
from sklearn import metrics as mt
from matplotlib import pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC ## Criando o Conjunto de Dados

# COMMAND ----------

X, _ = ds.make_blobs( n_samples=300, centers=4, cluster_std=0.6, random_state=0 )

plt.scatter( X[:, 0], X[:, 1], alpha=0.7, edgecolor='b' );

# COMMAND ----------

# MAGIC %md
# MAGIC ## Treinamento do Algoritmo

# COMMAND ----------

preferences = np.arange( -1, -50, -1 )
ss_list = list()

for i in range( len ( preferences ) ):
    model = c.AffinityPropagation( preference = preferences[i] )

    model.fit( X )
    labels = model.predict( X )

    ss = mt.silhouette_score( X, labels )
    ss_list.append(ss)

# COMMAND ----------

plt.plot( preferences, ss_list )
plt.xlabel( "Preference" )
plt.ylabel( "Silhoutte Score" )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modelo Final

# COMMAND ----------

p_best = ss_list.index( max( ss_list ) )

final_model = c.AffinityPropagation( preference=preferences[p_best] )
final_model.fit( X )
final_labels = final_model.predict( X )
final_ss = mt.silhouette_score( X, final_labels )

print( f"Número de Clusters = {len( np.unique( final_labels ) )}")
print( f"Final Silhouette Score = {final_ss}" )

plt.scatter( X[:, 0], X[:, 1], c=labels, cmap='rainbow', alpha=0.7, edgecolor='b' );
