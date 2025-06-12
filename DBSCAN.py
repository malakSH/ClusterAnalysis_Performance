# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 02:05:44 2024

@author: malak

Note: 
    The following is the steps and computational core of DBSCAN clustering built-in lib in python:
        * taken from sklearn module dbscan(): https://atavory.github.io/ibex/_modules/sklearn/cluster/dbscan_.html
        * For an example, see :ref:`examples/cluster/plot_dbscan.py
            <sphx_glr_auto_examples_cluster_plot_dbscan.py>`.
            Sparse neighborhoods can be precomputed using
            :func:`NearestNeighbors.radius_neighbors_graph
            <sklearn.neighbors.NearestNeighbors.radius_neighbors_graph>`
            with ``mode='distance'``.
        * Computational core (DOI of the paper WILL BE ADDED):
            # Calculate neighborhood for all samples. This leaves the original point
            # in, which needs to be considered later (i.e. point i is in the
            # neighborhood of point i. While True, its useless information)
            if metric == 'precomputed' and sparse.issparse(X):
                neighborhoods = np.empty(X.shape[0], dtype=object)
                X.sum_duplicates()  # XXX: modifies X's internals in-place
                X_mask = X.data <= eps
                masked_indices = X.indices.astype(np.intp, copy=False)[X_mask]
                masked_indptr = np.concatenate(([0], np.cumsum(X_mask)))[X.indptr[1:]]

            # insert the diagonal: a point is its own neighbor, but 0 distance
            # means absence from sparse matrix data
            masked_indices = np.insert(masked_indices, masked_indptr,
                                   np.arange(X.shape[0]))
            masked_indptr = masked_indptr[:-1] + np.arange(1, X.shape[0])
            # split into rows
            neighborhoods[:] = np.split(masked_indices, masked_indptr)
            else:
                neighbors_model = NearestNeighbors(radius=eps, algorithm=algorithm,
                                           leaf_size=leaf_size,
                                           metric=metric,
                                           metric_params=metric_params, p=p,
                                           n_jobs=n_jobs)
                neighbors_model.fit(X)
        # This has worst case O(n^2) memory complexity
        neighborhoods = neighbors_model.radius_neighbors(X, eps,
                                                         return_distance=False)

        if sample_weight is None:
            n_neighbors = np.array([len(neighbors)
                                for neighbors in neighborhoods])
        else:
            n_neighbors = np.array([np.sum(sample_weight[neighbors])
                                    for neighbors in neighborhoods])

        # Initially, all samples are noise.
        labels = -np.ones(X.shape[0], dtype=np.intp)

        # A list of all core samples found.
        core_samples = np.asarray(n_neighbors >= min_samples, dtype=np.uint8)
        dbscan_inner(core_samples, neighborhoods, labels)
        return np.where(core_samples)[0], labels
"""

import time
from sklearn.cluster import DBSCAN
import numpy as np

# Load synthetic data
synthetic_data = np.loadtxt('synthetic_data.csv', delimiter=',')
start_time = time.time()

# Initialize DBSCAN model
dbscan = DBSCAN(eps=0.5, min_samples=4)

# Measure the time taken for fitting the model
dbscan.fit(synthetic_data)

# Predict cluster labels
cluster_labels = dbscan.labels_

end_time = time.time()
processing_time = (end_time - start_time) * 1000


print("Processing Time:", processing_time, "milliseconds")

'''
import numpy as np
import time
from sklearn.metrics import silhouette_score

from sklearn.model_selection import ParameterGrid

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        self.labels_ = np.zeros(len(X))
        self.visited = set()

        current_label = 0

        for i in range(len(X)):
            if i in self.visited:
                continue

            self.visited.add(i)

            neighbors = self._find_neighbors(X, i)

            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1
            else:
                current_label += 1
                self.labels_[i] = current_label
                self._expand_cluster(X, neighbors, current_label)

        return self.labels_

    def _find_neighbors(self, X, index):
        neighbors = []
        for i in range(len(X)):
            if np.linalg.norm(X[index] - X[i]) < self.eps:
                neighbors.append(i)
        return neighbors

    def _expand_cluster(self, X, neighbors, current_label):
        i = 0
        while i < len(neighbors):
            neighbor_index = neighbors[i]
            if neighbor_index not in self.visited:
                self.visited.add(neighbor_index)
                new_neighbors = self._find_neighbors(X, neighbor_index)
                if len(new_neighbors) >= self.min_samples:
                    neighbors.extend(new_neighbors)
            if self.labels_[neighbor_index] == 0:
                self.labels_[neighbor_index] = current_label
            i += 1

# Load data from synthetic_data.csv
data = np.loadtxt('synthetic_data.csv', delimiter=',')



best_silhouette_score = -1
best_params = {}

# Define parameter grid for grid search
param_grid = {
    'eps': [0.1, 0.5, 1.0],
    'min_samples': [3, 5, 10]
}

# Perform grid search
for params in ParameterGrid(param_grid):
    dbscan = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
    labels = dbscan.fit(data)
    
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        silhouette_avg = silhouette_score(data, labels)
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_params = params

# Measure processing time
start_time = time.time()

# Initialize and fit DBSCAN with best parameters
dbscan = DBSCAN(eps=best_params['eps'], min_samples=best_params['min_samples'])
labels = dbscan.fit(data)

end_time = time.time()
processing_time = (end_time - start_time) * 1000  # in milliseconds


print("Best Parameters:", best_params)
print("Cluster Labels:", labels)
print("Processing Time:", processing_time, "milliseconds")
'''
