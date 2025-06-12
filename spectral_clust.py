# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:45:31 2024

@author: malak

Note: 
    The following is the steps and computational core of spectral clustering built-in lib in python:
        * taken from sklearn module class SpectralClustering() and function SpectralClustering(): https://atavory.github.io/ibex/_modules/sklearn/cluster/spectral.html
        * def __init__(self, n_clusters=8, eigen_solver=None, random_state=None,
                 n_init=10, gamma=1., affinity='rbf', n_neighbors=10,
                 eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1,
                 kernel_params=None, n_jobs=1):
            self.n_clusters = n_clusters
            self.eigen_solver = eigen_solver
            self.random_state = random_state
            self.n_init = n_init
            self.gamma = gamma
            self.affinity = affinity
            self.n_neighbors = n_neighbors
            self.eigen_tol = eigen_tol
            self.assign_labels = assign_labels
            self.degree = degree
            self.coef0 = coef0
            self.kernel_params = kernel_params
            self.n_jobs = n_jobs
            
            def fit(self, X, y=None):
                Creates an affinity matrix for X using the selected affinity,
                then applies spectral clustering to this affinity matrix.

                Parameters
                ----------
                X : array-like or sparse matrix, shape (n_samples, n_features)
                OR, if affinity==`precomputed`, a precomputed affinity
                matrix of shape (n_samples, n_samples)
                X = check_array(X, accept_sparse=['csr', 'csc', 'coo'],
                        dtype=np.float64)
                if X.shape[0] == X.shape[1] and self.affinity != "precomputed":
                    warnings.warn("The spectral clustering API has changed. ``fit``"
                                  "now constructs an affinity matrix from data. To use"
                                  " a custom affinity matrix, "
                                  "set ``affinity=precomputed``.")

            if self.affinity == 'nearest_neighbors':
            connectivity = kneighbors_graph(X, n_neighbors=self.n_neighbors, include_self=True,
                                            n_jobs=self.n_jobs)
            self.affinity_matrix_ = 0.5 * (connectivity + connectivity.T)
            elif self.affinity == 'precomputed':
                self.affinity_matrix_ = X
                else:
                    params = self.kernel_params
                    if params is None:
                        params = {}
                    if not callable(self.affinity):
                        params['gamma'] = self.gamma
                        params['degree'] = self.degree
                        params['coef0'] = self.coef0
                    self.affinity_matrix_ = pairwise_kernels(X, metric=self.affinity,
                                                     filter_params=True,
                                                     **params)

            random_state = check_random_state(self.random_state)
            self.labels_ = spectral_clustering(self.affinity_matrix_,
                                           n_clusters=self.n_clusters,
                                           eigen_solver=self.eigen_solver,
                                           random_state=random_state,
                                           n_init=self.n_init,
                                           eigen_tol=self.eigen_tol,
                                           assign_labels=self.assign_labels)
            return self

    * assign_labels : {'kmeans', 'discretize'}, default: 'kmeans'
        The strategy to use to assign labels in the embedding
        space. There are two ways to assign labels after the laplacian
        embedding. k-means can be applied and is a popular choice. But it can
        also be sensitive to initialization. Discretization is another approach
        which is less sensitive to random initialization.
        
        if assign_labels not in ('kmeans', 'discretize'):
            raise ValueError("The 'assign_labels' parameter should be "
                     "'kmeans' or 'discretize', but '%s' was given"
                     % assign_labels)

        random_state = check_random_state(random_state)
        n_components = n_clusters if n_components is None else n_components
        maps = spectral_embedding(affinity, n_components=n_components,
                          eigen_solver=eigen_solver,
                          random_state=random_state,
                          eigen_tol=eigen_tol, drop_first=False)

        if assign_labels == 'kmeans':
            _, labels, _ = k_means(maps, n_clusters, random_state=random_state,
                           n_init=n_init)
        else:
            labels = discretize(maps, random_state=random_state)

    return labels
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
import pandas as pd

# Read synthetic data from CSV file
synthetic_data = pd.read_csv('synthetic_data.csv', header=None).values

# Apply PCA for dimensionality reduction if needed
#pca = PCA(n_components=2)
#synthetic_data_pca = pca.fit_transform(synthetic_data)

# Perform spectral clustering
num_clusters = 22
spectral_clustering = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors')
cluster_labels = spectral_clustering.fit_predict(synthetic_data)

# Measure the time taken for fitting the model
start_time = time.time()
spectral_clustering.fit(synthetic_data)
end_time = time.time()
processing_time = (end_time - start_time)*1000

# Calculate silhouette score
silhouette_avg = silhouette_score(synthetic_data, cluster_labels)
print("Silhouette Score:", silhouette_avg)
print("Processing Time:", processing_time, "milliseconds")

# Plot the clusters (for 2D data)
# plt.scatter(synthetic_data_pca[:, 0], synthetic_data_pca[:, 1], c=cluster_labels, cmap='viridis')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.title('Spectral Clustering')
# plt.show()
