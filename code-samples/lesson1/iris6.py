import matplotlib.pyplot as plt
from sklearn import datasets


X_FEATURE = 0
IMG_PATH = "../../presentations/1. clustering and visualization/img/"

iris=datasets.load_iris()

plt.figure(figsize=(13, 4))
for subplot, y_feature in zip(range(1,4), range(1,4)):
    plt.subplot(1, 3, subplot) # nrows, ncols, index
    for target in set(iris.target):
        x = [iris.data[i,X_FEATURE] for i in range(len(iris.target)) if iris.target[i]==target]
        y = [iris.data[i,y_feature] for i in range(len(iris.target)) if iris.target[i]==target]
        plt.scatter(x, y, color=['red', 'blue', 'green'][target], label=iris.target_names[target])
    plt.xlabel(iris.feature_names[X_FEATURE])
    plt.ylabel(iris.feature_names[y_feature])
plt.legend(iris.target_names, loc='lower right')
plt.savefig(IMG_PATH + iris.feature_names[X_FEATURE] + ".png")
