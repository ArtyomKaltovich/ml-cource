import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering
from sklearn import preprocessing


X_FEATURE = 2
Y_FEATURE = 3

class ClusterDrawer:
    row = 1
    index = 1

    def __init__(self, data, row=1):
        self.row = row
        self.data = data

    def draw_and_predict(self, clf, title):
        self.draw(self.data.target, 'True')
        # plt.scatter(x, y, color=['red', 'blue', 'green'][target], label=iris.target_names[target])
        plt.legend(self.data.target_names, loc='lower right')
        pred = clf.fit_predict(self.data.data)
        self.draw(pred, title)

    def draw(self, clazz, title):
        plt.subplot(self.row, 2, self.index)  # nrows, ncols, index
        plt.title(title)
        self.index += 1
        for target in set(clazz):
            x = [self.data.data[i, X_FEATURE] for i in range(len(clazz)) if clazz[i] == target]
            y = [self.data.data[i, Y_FEATURE] for i in range(len(clazz)) if clazz[i] == target]
            plt.scatter(x, y)
        plt.xlabel(self.data.feature_names[X_FEATURE])
        plt.ylabel(self.data.feature_names[Y_FEATURE])

    def show(self):
        plt.tight_layout(h_pad=-2)
        plt.show()


iris = datasets.load_iris()
#iris.data = preprocessing.scale(iris.data)
drawer = ClusterDrawer(row=3, data=iris)
clf = KMeans(n_clusters=3)
drawer.draw_and_predict(clf, 'K-Means')

clf = AffinityPropagation(preference=-50)
drawer.draw_and_predict(clf, 'Affinity')

clf = AgglomerativeClustering(n_clusters=3)
drawer.draw_and_predict(clf, 'Agglomerative')

drawer.show()
