# Databricks notebook source
# MAGIC %md
# MAGIC # Clusterização com K-Means

# COMMAND ----------

from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ## Criando o conjunto de dados

# COMMAND ----------

# Criando Dados Sintéticos
X, y = make_blobs (
    n_samples = 100,
    n_features = 2,
    centers = 4,
    cluster_std = 0.9,
    random_state = 0)

# COMMAND ----------

# Visualizando o conjunto de dados
plt.scatter( X[:, 0], X[:, 1] );

# COMMAND ----------

# MAGIC %md
# MAGIC ## Treinando o modelo

# COMMAND ----------

# Treinando o modelo
k = 2

kmeans = KMeans(
        n_clusters = k, 
        init = 'random', 
        n_init = 10,
        random_state = 0)

labels = kmeans.fit_predict( X )

# COMMAND ----------

# Verificando a performance do modelo
silhouette_avg = silhouette_score( X, labels )

print(f"For {k} clusters the average Silhouette Score is {np.round(silhouette_avg, 2)}")

# Visualizando a clusterização do algoritmo
plt.scatter( X[:, 0], X[:, 1], c = kmeans.labels_ );

# COMMAND ----------

# Treinando o modelo
k2 = 4

kmeans2 = KMeans(
        n_clusters = k2, 
        init = 'random', 
        n_init = 10,
        random_state = 0)

labels2 = kmeans2.fit_predict( X )

# COMMAND ----------

# Verificando a performance do modelo
silhouette_avg2 = silhouette_score( X, labels2 )

print(f"For {k2} clusters the average Silhouette Score is {np.round(silhouette_avg2, 2)}")

# Visualizando a clusterização do algoritmo
plt.scatter( X[:, 0], X[:, 1], c = kmeans2.labels_ )

# COMMAND ----------

n_k = np.arange( 2, 11, 1 )
score_list = list()

for n in n_k:
    
    k_means = KMeans(
                n_clusters= n, 
                init= 'random', 
                n_init= 10,
                random_state= 0)
    
    labels_loop = k_means.fit_predict( X )

    ss_loop = silhouette_score( X, labels_loop )
    score_list.append( ss_loop )

    print(f"For {n} clusters the average Silhouette Score is {np.round( ss_loop, 2 )}")

plt.plot( n_k, score_list, marker = 'o' );
