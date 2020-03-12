#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 09:13:15 2020
WPI CS539 machine learning Spring 2020
team assignment 2 problem 1
used example code from scikit learn as starting point
https://scikit-learn.org/stable/auto_examples/cluster/plot_linkage_comparison.html#sphx-glr-auto-examples-cluster-plot-linkage-comparison-py
"""

import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import homogeneity_score
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import adjusted_mutual_info_score as AMI
from itertools import cycle, islice

np.random.seed(0)

######################################################################
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times

n_samples = 1000
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.6, noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)

######################################################################
# Run the clustering and plot

# Set up cluster parameters
plt.figure(1, figsize=(10 * 1.3 + 2, 15.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)
plt.figure(2, figsize=(10 * 1.3 + 2, 15.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)


plot1_num = 1
plot2_num = 1

default_base = {'n_neighbors': 10,
                'n_clusters': 2}

datasets = [
    (noisy_circles, {'n_clusters': 2}),
    (noisy_moons, {'n_clusters': 2})]

for i_dataset, (dataset, algo_params) in enumerate(datasets):
    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)

    X, y = dataset

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # ============
    # Create cluster objects
    # ============
    kmeans = cluster.KMeans(n_clusters=params['n_clusters'])
    kmeans_rand = cluster.KMeans(n_clusters=params['n_clusters'], init='random')
    kmeans_5 = cluster.KMeans(n_clusters=params['n_clusters'], n_init=5)
    kmeans_20 = cluster.KMeans(n_clusters=params['n_clusters'], n_init=20, max_iter=500)
    complete = cluster.AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='complete')
    average = cluster.AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='average')
    single = cluster.AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='single')

    kmeans_variations = (
        ('KMeans_k++', kmeans),
        ('KMeans_random', kmeans_rand),
        ('KMeans_5seeds', kmeans_5),
        ('KMeans_20seeds', kmeans_20)
    )

    clustering_algorithms = (
        ('KMeans', kmeans),
        ('Single Linkage', single),
        ('Average Linkage', average),
        ('Complete Linkage', complete)
    )

    # Compare KMeans initialization
    for name, algorithm in kmeans_variations:
        t0 = time.time()

        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the " +
                "connectivity matrix is [0-9]{1,2}" +
                " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            algorithm.fit(X)

        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

        plt.figure(1)
        plt.subplot(len(datasets), len(kmeans_variations), plot1_num)
        if i_dataset == 0:
            plt.title(name, size=10)

        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))

        plt.scatter(X[:, 0], X[:, 1], s=5, color=colors[y_pred])
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('Comp time ' '%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')
        plt.text(.99, .06, ('Purity score ' '%.3f' % (homogeneity_score(y, y_pred))).lstrip('0'),
                 transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')
        plt.text(.99, .11, ('Rand Index ' '%.3f' % (ARI(y, y_pred))).lstrip('0'),
                 transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')
        plt.text(.99, .16, ('Mutual Info ' '%.3f' % (AMI(y, y_pred))).lstrip('0'),
                 transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')
        plot1_num += 1

    # Compare Clustering Algorithms
    for name, algorithm in clustering_algorithms:
        t0 = time.time()

        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the " +
                        "connectivity matrix is [0-9]{1,2}" +
                        " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            algorithm.fit(X)

        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

        plt.figure(2)
        plt.subplot(len(datasets), len(clustering_algorithms), plot2_num)
        if i_dataset == 0:
            plt.title(name, size=10)

        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))

        plt.scatter(X[:, 0], X[:, 1], s=5, color=colors[y_pred])
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('Comp time ' '%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')
        plt.text(.99, .06, ('Purity score ' '%.3f' % (homogeneity_score(y, y_pred))).lstrip('0'),
                 transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')
        plt.text(.99, .11, ('Rand Index ' '%.3f' % (ARI(y, y_pred))).lstrip('0'),
                 transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')
        plt.text(.99, .16, ('Mutual Info ' '%.3f' % (AMI(y, y_pred))).lstrip('0'),
                 transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')
        plot2_num += 1

plt.show()
