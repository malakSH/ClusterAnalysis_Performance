# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 00:21:01 2024

@author: malak
"""

import numpy as np
from scipy.spatial.distance import euclidean
from collections import Counter
import pandas as pd
import time
class DensityPeakCluster:
    def __init__(self, delta, rho):
        self.delta = delta
        self.rho = rho
        self.cluster_centers = []

    def fit_predict(self, X):
        distances = self.compute_distances(X)
        self.rho_values = self.compute_rho(distances)
        self.delta_values = self.compute_delta(distances)
        self.cluster_centers = self.identify_cluster_centers()
        return self.assign_clusters(distances)

    def compute_distances(self, X):
        n = len(X)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                distances[i, j] = euclidean(X[i], X[j])
                distances[j, i] = distances[i, j]  # Symmetric matrix
        return distances

    def compute_rho(self, distances):
        n = distances.shape[0]
        rho = np.zeros(n)
        for i in range(n):
            rho[i] = np.sum(distances[i] < self.delta)
        return rho

    def compute_delta(self, distances):
        n = distances.shape[0]
        delta = np.zeros(n)
        sorted_rho_indices = np.argsort(-self.rho_values)  # Sort in descending order
        for i in range(1, n):
            idx = sorted_rho_indices[i]
            delta[idx] = np.min(distances[idx, sorted_rho_indices[:i]])
        return delta

    def identify_cluster_centers(self):
        cluster_centers = []
        for i, rho_i in enumerate(self.rho_values):
            if rho_i > np.max(self.rho_values) * 0.1:  # Adjust threshold as needed
                if self.delta_values[i] > np.median(self.delta_values):
                    cluster_centers.append(i)
        return cluster_centers

    def assign_clusters(self, distances):
        n = distances.shape[0]
        clusters = np.zeros(n, dtype=int)
        for i in range(n):
            if i in self.cluster_centers:
                clusters[i] = self.cluster_centers.index(i) + 1
            else:
                nearest_center = np.argmin(distances[i, self.cluster_centers])
                clusters[i] = nearest_center + 1
        return clusters

def ensemble_density_peak_clustering(X, delta_values, rho_values, num_clusterings):
    clusterings = []
    for _ in range(num_clusterings):
        delta = np.random.choice(delta_values)
        rho = np.random.choice(rho_values)
        dpc = DensityPeakCluster(delta=delta, rho=rho)
        clusters = dpc.fit_predict(X)
        clusterings.append(clusters)
    return combine_clusterings(clusterings)

def combine_clusterings(clusterings):
    # Combine clusterings using majority voting
    ensemble_clusters = np.array([Counter([c[i] for c in clusterings]).most_common(1)[0][0] for i in range(len(clusterings[0]))])
    return ensemble_clusters

# Read data from CSV file
data = pd.read_csv("synthetic_data.csv")
X = data.values  # Assuming no target label

# parameters for Density Peak clustering
delta_values = [0.1, 0.5, 1.0]
rho_values = [2, 5, 10]
#num_clusterings = 5
delta_values = [0.1, 0.5, 1.0]
rho_values = [2, 5, 10]
num_clusterings = 22
start_time = time.time()
ensemble_clusters = ensemble_density_peak_clustering(X, delta_values, rho_values, num_clusterings)
#print("Ensemble Clusters:", ensemble_clusters)
# Measure the time taken for fitting the model
end_time = time.time()
processing_time = (end_time - start_time)*1000

print("Processing Time:", processing_time, "milliseconds")
