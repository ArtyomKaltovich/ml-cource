import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing


X_FEATURE = 2
Y_FEATURE = 3
MESH_STEP_SIZE = 0.05

iris = datasets.load_iris()
#iris.data = preprocessing.scale(iris.data)

def draw(index, title, X_train=iris.data, y_train=iris.target, X_test=None, y_test=None):
    plt.subplot(2, 2, index)  # nrows, ncols, index
    plt.title(title)
    draw_scatter(X_train, y_train, marker='o')
    if X_test is not None:
        draw_scatter(X_test, y_test, marker='x')
    plt.xlabel(iris.feature_names[X_FEATURE])
    plt.ylabel(iris.feature_names[Y_FEATURE])


def draw_scatter(X_train, y_train, marker):
    for target in set(y_train):
        x = [X_train[i, X_FEATURE] for i in range(len(y_train)) if y_train[i] == target]
        y = [X_train[i, Y_FEATURE] for i in range(len(y_train)) if y_train[i] == target]
        plt.scatter(x, y, marker=marker, color=['red', 'green', 'blue'][target])


draw(1, 'True')
print(iris.data)

clf = KMeans(n_clusters=3)
pred = clf.fit_predict(iris.data)
draw(2, 'K-means', y_train=pred)

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33)

clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train, y_train)
train_pred = clf.predict(X_train)
test_pred = clf.predict(X_test)
draw(3, 'Neighbors k=1 accuracy=%.3f' % accuracy_score(y_test, test_pred),
     X_train = X_train, y_train=train_pred, X_test = X_test, y_test=test_pred)

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)
train_pred = clf.predict(X_train)
test_pred = clf.predict(X_test)
draw(4, 'Neighbors k=5 accuracy=%.3f' % accuracy_score(y_test, test_pred),
     X_train = X_train, y_train=train_pred, X_test = X_test, y_test=test_pred)

plt.tight_layout(h_pad=-2)
plt.show()
