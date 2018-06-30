from pandas import read_csv, DataFrame
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from numpy import mean
import os, sys
PACKAGE_PARENT = '../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from utils.lib import extract_target_column


def prepare(path):
    df = read_csv(path, index_col='PassengerId')
    df['is_male'] = df['Sex'] == 'male'
    #df['is_female'] = df['Sex'] == 'female'
    df['to_Cherbourg'] = df['Embarked'] == 'C'
    df['to_Queenstown'] = df['Embarked'] == 'Q'
    df['to_Southampton'] = df['Embarked'] == 'S'
    for clazz in df['Pclass'].unique():
        df['is_%d_class' % clazz] = df['Pclass'] == clazz
    #df['unknown_embarked'] = df['Embarked'].isnull()
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Fare'] = df['Fare'].fillna(0)
    df = df.drop(['Ticket', 'Cabin', 'Name', 'Embarked', 'Sex', 'Pclass'], axis=1)
    return df


class TitanicClassifier(BaseEstimator):
    def fit(self, X, y):
        pass

    def predict(self, y):
        return (y['is_male'] == False)


train = prepare('data/train.csv')
print(len(train))
new_train = train[train['Survived'] == True]
new_train = new_train.append(train[train['Survived'] == False].sample(len(new_train)))
print(len(new_train))
X, y = extract_target_column(new_train, 'Survived')
test = prepare('data/test.csv')

cv = KFold(n_splits=5, shuffle=True)
clf = TitanicClassifier()
acc = cross_val_score(clf, X, y, scoring='accuracy', cv=cv)
print(min(acc), mean(acc))

clf = DecisionTreeClassifier()
acc = cross_val_score(clf, X, y, scoring='accuracy', cv=cv)
print(min(acc), mean(acc))

parameters = { 'criterion': ('gini', 'entropy'),
               'min_samples_split': (2,3,5,10,15,20,25,30,40,50,60,75,85,100),
               'min_samples_leaf':  (1, 2,3,5,10,15,20,25)
             }
clf = DecisionTreeClassifier()
clf = GridSearchCV(clf, parameters, n_jobs=-1)
clf.fit(X, y)
print(clf.best_params_)
print(clf.best_score_)

clf = clf.best_estimator_
clf.fit(X, y)
pred = clf.predict(test)
answer = DataFrame(index=test.index)
answer['Survived'] = pred
answer.to_csv('result/result.csv')
export_graphviz(clf, out_file='result/tree.dot', feature_names=X.columns)
