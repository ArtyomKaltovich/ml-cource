from pandas import read_csv, DataFrame
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
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

train = prepare('data/train.csv')
X, y = extract_target_column(train, 'Survived')
test = prepare('data/test.csv')
X = X / X.max()
test = test / test.max()

cv = KFold(n_splits=3, shuffle=True)
clf = SVC()
acc = cross_val_score(clf, X, y, scoring='accuracy', cv=cv)
print(min(acc), mean(acc))

parameters = [{ 'kernel': ('linear', 'rbf', 'sigmoid'),
               #'C': (0.33, 1.0, 3.3, 10, 33),
               #'gamma':  ('auto', 0, 0.01, 0.33, 1.0),
               #'cache_size': (1000,),
               #'class_weight': ('balanced', None),
             }]
parameters= [{ 'kernel': ('poly',),
                'C': (1.0, 3.3, 10),
                'gamma':  ('auto', 0.33, 1.0, 3.3),
                #'degree':  (2, 3, 5),
                #'coef0':  (0.0, 0.01, 0.1),
                'cache_size': (1000,),
                'class_weight': ('balanced', None)
              }]
clf = SVC()
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
