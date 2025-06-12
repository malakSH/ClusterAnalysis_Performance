# -*- coding: utf-8 -*-
"""
Created on Fri May  3 22:04:12 2024

@author: malak

Note: 
    The following is the steps and computational core of k-means clustering built-in lib in python:
        * taken from sklearn module _kmeans_single_lloyd(): https://atavory.github.io/ibex/_modules/sklearn/cluster/k_means_.html
        * A single run of k-means, assumes preparation completed prior.
            Parameters
            ----------
            X : array-like of floats, shape (n_samples, n_features) 
            The observations to cluster.

            n_clusters : int
            The number of clusters to form as well as the number of
            centroids to generate.

            max_iter : int, optional, default 300
            Maximum number of iterations of the k-means algorithm to run.

            init : {'k-means++', 'random', or ndarray, or a callable}, optional
            Method for initialization, default to 'k-means++':

                'k-means++' : selects initial cluster centers for k-mean
                clustering in a smart way to speed up convergence. See section
                Notes in k_init for more details.

                'random': generate k centroids from a Gaussian with mean and
                variance estimated from the data.

                    If an ndarray is passed, it should be of shape (k, p) and gives
                    the initial centers.

                    If a callable is passed, it should take arguments X, k and
                    and a random state and return an initialization.

            tol : float, optional
            The relative increment in the results before declaring convergence.

            verbose : boolean, optional
            Verbosity mode

            x_squared_norms : array
            Precomputed x_squared_norms.

            precompute_distances : boolean, default: True
            Precompute distances (faster but takes more memory).

            random_state : int, RandomState instance or None, optional, default: None
                If int, random_state is the seed used by the random number generator;
                If RandomState instance, random_state is the random number generator;
                If None, the random number generator is the RandomState instance used
                by `np.random`.

            Returns
            -------
            centroid : float ndarray with shape (k, n_features)
            Centroids found at the last iteration of k-means.

            label : integer ndarray with shape (n_samples,)
                label[i] is the code or index of the centroid the
                i'th observation is closest to.

            inertia : float
                The final value of the inertia criterion (sum of squared distances to 
                the closest centroid for all observations in the training set).

                n_iter : int
                Number of iterations run.
    
        * computational core: DOI of the paper WILL BE ADDED
            for i in range(max_iter):
                lloyd_iter(
                    X,
                    sample_weight,
                    centers,
                    centers_new,
                    weight_in_clusters,
                    labels,
                    center_shift,
                    n_threads,
                    )

                if verbose:
                    inertia = _inertia(X, sample_weight, centers, labels, n_threads)
                    print(f"Iteration {i}, inertia {inertia}.")

                centers, centers_new = centers_new, centers

                # Check for convergence
                if np.array_equal(labels, labels_old):
                    if verbose:
                        print(f"Converged at iteration {i}: strict convergence.")
                        strict_convergence = True
                        break
                    else:
                        center_shift_tot = (center_shift**2).sum()
                        if center_shift_tot <= tol:
                            if verbose:
                                print(f"Converged at iteration {i}: center shift {center_shift_tot} within tolerance {tol}.")
                                break

                labels_old[:] = labels
        
"""

import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# Scenario 1: 100 instances, 2 clusters
data_scenario1 = np.loadtxt('synthetic_data.csv', delimiter=',')
num_clusters_scenario1 = 2

# Initialize KMeans model 
kmeans_scenario1 = KMeans(n_clusters=num_clusters_scenario1, random_state=0)

# Measure the time taken 
start_time_scenario1 = time.time()
kmeans_scenario1.fit(data_scenario1)
end_time_scenario1 = time.time()
processing_time_scenario1 = (end_time_scenario1 - start_time_scenario1) * 1000

# Predict cluster labels 
cluster_labels_scenario1 = kmeans_scenario1.predict(data_scenario1)

# Compute silhouette score 
silhouette_avg_scenario1 = silhouette_score(data_scenario1, cluster_labels_scenario1)

print("Scenario 1 - Processing Time:", processing_time_scenario1, "milliseconds")
print("Scenario 1 - Silhouette Score:", silhouette_avg_scenario1)
