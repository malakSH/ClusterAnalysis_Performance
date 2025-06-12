# -*- coding: utf-8 -*-
"""
Created on Mon May  6 20:57:45 2024

@author: malak

Note: 
    The following is the steps and computational core of BIRCH clustering built-in lib in python:
        * taken from sklearn module https://atavory.github.io/ibex/_modules/sklearn/cluster/birch.html
        * Computational core (DOI of the paper WILL BE ADDED):
            Processing Data Points:
                For each data point, the closest centroid is found, and the CF (Clustering Feature) is updated. 
                This is handled in the insert_cf_subcluster method of the _CFNode class
            
            When the branching factor is reached, the node is split into two new nodes 
            This is handled in the _split_node function
            
            Updating CF and Radius Calculation is handled in class _CFSubcluster()
            
            Then, the global clustering function:
                def _global_clustering(self, X=None):
                    clusterer = self.n_clusters
                    centroids = self.subcluster_centers_
                    compute_labels = (X is not None) and self.compute_labels

            # Preprocessing for the global clustering.
            not_enough_centroids = False
            if isinstance(clusterer, int):
                clusterer = AgglomerativeClustering(
                    n_clusters=self.n_clusters)
                # There is no need to perform the global clustering step.
                if len(centroids) < self.n_clusters:
                not_enough_centroids = True
                elif (clusterer is not None and not
                      hasattr(clusterer, 'fit_predict')):
                    raise ValueError("n_clusters should be an instance of "
                                     "ClusterMixin or an int")

            # To use in predict to avoid recalculation.
            self._subcluster_norms = row_norms(
                self.subcluster_centers_, squared=True)

            if clusterer is None or not_enough_centroids:
                self.subcluster_labels_ = np.arange(len(centroids))
               if not_enough_centroids:
                   warnings.warn(
                       "Number of subclusters found (%d) by Birch is less "
                       "than (%d). Decrease the threshold."
                       % (len(centroids), self.n_clusters))
                   else:
                       # The global clustering step that clusters the subclusters of
                       # the leaves. It assumes the centroids of the subclusters as
                       # samples and finds the final centroids.
                       self.subcluster_labels_ = clusterer.fit_predict(
                           self.subcluster_centers_)

            if compute_labels:
                self.labels_ = self.predict(X)
          
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score
import time

# Load your CSV file
file_path = 'synthetic_data4.csv'
data = pd.read_csv(file_path)

start_time = time.time()

# Extract the features you want to use for clustering
features = data.iloc[:, [0, 1]].values

# Creating the BIRCH clustering model
# Adjust parameters as needed
model = Birch(branching_factor=2, n_clusters=8, threshold=0.25)

# Fit the data (Training)
model.fit(features)


# Predict the clusters for the data
pred = model.predict(features)

# Measure the end time
end_time = time.time()

# Calculate the processing time in milliseconds
processing_time_ms = (end_time - start_time) * 1000
#1000000000 
#1000
print("Processing Time (ms):", processing_time_ms)

# Compute the silhouette score
silhouette_avg = silhouette_score(features, pred)
print("Silhouette Score:", silhouette_avg)

# Creating a scatter plot
#plt.scatter(features[:, 0], features[:, 1], c=pred, cmap='rainbow', alpha=0.7, edgecolors='b')
#plt.show()
