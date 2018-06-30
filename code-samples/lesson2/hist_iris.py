import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestCentroid
import numpy

iris = datasets.load_iris()
distances = pdist(iris.data)
print(distances.shape)

clf = NearestCentroid(metric='euclidean')
clf.fit(iris.data, iris.target)
print(pdist(clf.centroids_))

plt.hist(distances, 100)
plt.show()