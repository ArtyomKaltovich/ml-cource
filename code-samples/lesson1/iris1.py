import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
plt.scatter(iris.data[:,0], iris.data[:,1])
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

plt.show()