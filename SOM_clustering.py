# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 06:24:58 2024

@author: malak
"""
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.metrics import silhouette_score

import time
# Read data from CSV file
data = pd.read_csv("synthetic_data11.csv").values
time_start = time.time()
# Parameters 
k = 22
alpha = 0.92 # Initial learning rate

# Add zero column to data
n, d = data.shape
data = np.append(data, np.zeros((n, 1)), axis=1)

# Print data information
'''
print("The training data: \n", data)
print("\nTotal number of data: ", n)
print("Total number of features: ", d)
print("Total number of Clusters: ", k)
'''
# Initialize weights
weights = np.random.rand(d, k)
#print("\nThe initial weight: \n", np.round(weights, 2))

# SOM algorithm
for it in range(1000):  # Total number of iterations
    for i in range(n):
        dist_min = np.inf
        for j in range(k):
            dist = np.square(distance.euclidean(weights[:, j], data[i, 0:d]))
            if dist_min > dist:
                dist_min = dist
                j_min = j
        weights[:, j_min] = weights[:, j_min] * (1 - alpha) + alpha * data[i, 0:d]
    alpha = 0.5 * alpha

#print("\nThe final weight: \n", np.round(weights, 4))

# Assign clusters
for i in range(n):
    c_number = np.argmax(np.dot(data[i, 0:d], weights))
    data[i, d] = c_number
time_end = time.time()

# Calculate the processing time in milliseconds
processing_time_ms = (time_end - time_start) * 1000
print("Processing Time (ms):", processing_time_ms)


#print("\nThe data with cluster number: \n", data)

# Assuming you have your clusters assigned in the last column of your data array
clusters = data[:, -1].astype(int)


# Compute silhouette score
silhouette_avg = silhouette_score(data[:, :-1], clusters)
print("The average silhouette score is:", silhouette_avg)
