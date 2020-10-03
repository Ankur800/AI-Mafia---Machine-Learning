import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.10)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# Can't seperate out
# let's try
from sklearn.cluster import KMeans
km = KMeans(n_clusters=2)
km.fit(X)

centers = km.cluster_centers_
label = km.labels_

plt.scatter(X[:, 0], X[:, 1], c=label)
plt.scatter(centers[:, 0], centers[:, 1], color='red')
plt.show()

# We can see the plotting is not good as it should be
# DENSITY BASED CLUSTERING
from sklearn.cluster import DBSCAN
dbs = DBSCAN(eps=0.214, min_samples=5)
dbs.fit(X)

y_pred = dbs.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()