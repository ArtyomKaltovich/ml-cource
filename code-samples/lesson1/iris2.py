import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
plt.scatter(iris.data[:,2], iris.data[:,3])
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])

plt.show()