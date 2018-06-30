import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing


X_FEATURE = 2
Y_FEATURE = 3
MESH_STEP_SIZE = 0.05

def draw(clazz, index, title):
    plt.subplot(2, 2, index)  # nrows, ncols, index
    plt.title(title)
    for target in set(clazz):
        x = [iris.data[i, X_FEATURE] for i in range(len(clazz)) if clazz[i] == target]
        y = [iris.data[i, Y_FEATURE] for i in range(len(clazz)) if clazz[i] == target]
        plt.scatter(x, y)
    plt.xlabel(iris.feature_names[X_FEATURE])
    plt.ylabel(iris.feature_names[Y_FEATURE])


iris = datasets.load_iris()
draw(iris.target, 1, 'True')

clf = KMeans(n_clusters=3)
pred = clf.fit_predict(iris.data)
draw(pred, 2, 'K-means')

clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(iris.data, iris.target)
pred = clf.predict(iris.data)
draw(pred, 3, 'Neighbors k=1')

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(iris.data, iris.target)
pred = clf.predict(iris.data)
draw(pred, 4, 'Neighbors k=5')

#plt.tight_layout(h_pad=-2)
plt.show()
