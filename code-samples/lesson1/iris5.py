import matplotlib.pyplot as plt
from sklearn import datasets
from pandas import DataFrame

iris=datasets.load_iris()

df = DataFrame()

for i in range(len(iris.feature_names)):
    df[iris.feature_names[i]] = iris.data[:, i]
df["target"] = iris.target

print(df.head())

plt.matshow(df.corr())
plt.xticks(range(len(df.columns)), df.columns, rotation=30)
plt.yticks(range(len(df.columns)), df.columns, rotation=60)
plt.colorbar()
plt.show()