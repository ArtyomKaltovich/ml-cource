from pandas import read_csv, DataFrame
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
    df = df.drop(['Ticket', 'Cabin', 'Name', 'Embarked', 'Sex', 'Pclass'], axis=1)
    return df


train = prepare('data/train.csv')
# print(len(train))
# new_train = train[train['Survived'] == True]
# new_train = new_train.append(train[train['Survived'] == False].sample(len(new_train)))
# print(len(new_train))
X, y = extract_target_column(train, 'Survived')
test = prepare('data/test.csv')

cv = KFold(n_splits=3, shuffle=True)

clf = RandomForestClassifier()
acc = cross_val_score(clf, X, y, scoring='accuracy', cv=cv)
print(min(acc), mean(acc))

parameters = { 'criterion': ('gini', 'entropy'),
               'min_samples_split': (3, 10, 33, 100),
               'min_samples_leaf':  (1, 3, 10, 33),
               'n_estimators': (66, 100, 150),
               'class_weight': ('balanced', None),
               'max_features': (None, 'auto', 'sqrt', 'log2'),
               'max_depth': (33, 100, 333, None),
               'min_weight_fraction_leaf': (0.0, 0.01, 0.1),
               'max_leaf_nodes': (None, 10, 33, 100),
               'min_impurity_decrease': (0.0, 0.01, 0.1),
             }
#{'class_weight': None, 'criterion': 'gini', 'max_depth': 33, 'max_features': None, 'max_leaf_nodes': 33, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 10}
#0.8361391694725028
clf = RandomForestClassifier()
clf = GridSearchCV(clf, parameters, n_jobs=-1)
clf.fit(X, y)
print(clf.best_params_)
print(clf.best_score_)

clf = clf.best_estimator_
#clf.fit(X, y)
pred = clf.predict(test)
answer = DataFrame(index=test.index)
answer['Survived'] = pred
answer.to_csv('result/result.csv')
