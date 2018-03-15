import matplotlib.pyplot as plt
from sklearn import datasets

iris=datasets.load_iris()
X_FEATURE = 0
Y_FEATURE = 1

for target in set(iris.target):
    x = [iris.data[i,X_FEATURE] for i in range(len(iris.target)) if iris.target[i]==target]
    y = [iris.data[i,Y_FEATURE] for i in range(len(iris.target)) if iris.target[i]==target]
    plt.scatter(x, y, color=['red', 'blue', 'green'][target], label=iris.target_names[target])
plt.xlabel(iris.feature_names[X_FEATURE])
plt.ylabel(iris.feature_names[Y_FEATURE])
plt.title('Iris Dataset')
plt.legend(iris.target_names, loc='lower right')
plt.show()