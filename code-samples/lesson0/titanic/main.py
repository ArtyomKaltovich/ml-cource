from pandas import read_csv, DataFrame
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import accuracy_score
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
        return (y['is_male'] == False) | (y['Age'] < 18.0)
               #len(y) * [False]
               #y['is_1_class']
               #y['Age'] < 18.0
               #(y['is_male'] == False)


train = prepare('data/train.csv')
X, y = extract_target_column(train, 'Survived')
test = prepare('data/test.csv')
