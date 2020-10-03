# IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TAKING RANDOM DATA FROM SKLEARN
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=1000, n_features=2, centers=5, random_state=3)

plt.scatter(X[:, 0], X[:, 1])
plt.show()

k = 5
colors = ["green", "blue", "red", "yellow", "gray"]
clusters = {}

for each_cluster in range(k):
    # Step 1: To initialise cluster centers randomly
    center = 10 * (2 * np.random.random((X.shape[1],)) - 1)

    points = []

    cluster = {
        'center' : center,
        'points' : points,
        'color' : colors[each_cluster]
    }

    clusters[each_cluster] = cluster

# Step 2: To find out distance and assigning points to the clusters
def distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2)**2))

# Assigning every data point to one of the cluster
# E - Step - Expectation Step
def assignPointsToCluster(clusters):

    for every_data_point in range(X.shape[0]):

        dist = []
        current_X = X[every_data_point]
        for kx in range(k):
            d = distance(current_X, clusters[kx]['center'])
            dist.append(d)

        current_cluster = np.argmin(dist)
        clusters[current_cluster]['points'].append(current_X)

# M - Step - Maximization Step
# STEP - 3
# Update your cluster center by taking mean

def updateCluster(cluster):

    for kx in range(k):

        pts = np.array(clusters[kx]['points'])

        if pts.shape[0] > 0:

            # We will find out mean
            new_u = pts.mean(axis=0)
            clusters[kx]['center'] = new_u
            clusters[kx]['points'] = []

def plotClusters(clusters):

    for kx in range(k):

        pts = np.array(clusters[kx]['points'])

        if pts.shape[0] > 0:
            plt.scatter(pts[:, 0], pts[:, 1], color=clusters[kx]['color'])

        plt.scatter(clusters[kx]['center'][0], clusters[kx]['center'][1], color='black', marker='*')
    plt.show()

assignPointsToCluster(clusters)
plotClusters(clusters)
updateCluster(clusters)