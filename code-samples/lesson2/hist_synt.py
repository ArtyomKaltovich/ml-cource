import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestCentroid
from sklearn.datasets.samples_generator import make_blobs
import numpy

centers = [[1, 1], [-1, -1], [1, -1], [-1, 1]]
X, y = make_blobs(n_samples=300, centers=centers, cluster_std=0.2)

for target in set(y):
    x = [X[i, 0] for i in range(len(y)) if y[i] == target]
    yy = [X[i, 1] for i in range(len(y)) if y[i] == target]
    plt.scatter(x, yy)

plt.show()
plt.clf()

distances = pdist(X)

clf = NearestCentroid(metric='euclidean')
clf.fit(X, y)
print(pdist(clf.centroids_))

plt.hist(distances, 100)
plt.show()