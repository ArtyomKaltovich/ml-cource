from pandas import read_csv, DataFrame, concat
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import export_graphviz
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, train_test_split
from numpy import mean
import xgboost as xgb
import matplotlib.pyplot as plt
import os, sys

from xgboost import XGBClassifier

PACKAGE_PARENT = '../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from utils.lib import extract_target_column


def get_data(path_train, path_test):
    train = read_and_common_preprocess(path_train)
    test = read_and_common_preprocess(path_test)
    df = concat([train, test])
    for x in [train, test]:
        x['Age'] = x['Age'].fillna(df['Age'].median())
        x['Fare'] = x['Fare'].fillna(df['Fare'].median())
        x['ticket_size'] = df.groupby(['Ticket'])['Sex'].transform('count')
        x['is_alone1'] = x['ticket_size'] == 1
        #x['is_alone'] = (x['ticket_size'] == 1) | (df['family_size'] == 1) | (df['family_size'] == 0)
        x['fpp'] = x['Fare'] / x['ticket_size']
        #x['cabin_len'] = x['Cabin'].str.len()
        for clazz in df['Cabin'].str.slice(stop=1).unique():
            x['is_%s_cabin' % clazz] = x['Cabin'].str.slice(stop=1) == clazz


    train = train.drop(['Ticket', 'Cabin', 'Name', 'Embarked', 'Sex', 'Pclass'], axis=1)
    test = test.drop(['Ticket', 'Cabin', 'Name', 'Embarked', 'Sex', 'Pclass'], axis=1)
    #print(train.columns)
    return train, test


def read_and_common_preprocess(path):
    df = read_csv(path, index_col='PassengerId')
    df['is_male'] = df['Sex'] == 'male'
    df['age_null'] = df['Age'].isnull()
    df['cabin_null'] = df['Cabin'].isnull()
    # df['is_female'] = df['Sex'] == 'female'
    df['from_Cherbourg'] = df['Embarked'] == 'C'
    df['from_Queenstown'] = df['Embarked'] == 'Q'
    df['from_Southampton'] = df['Embarked'] == 'S'
    df['family_size'] = df['SibSp'] + df['Parch']
    df['is_alone2'] = df['family_size'] == 0
    df['is_alone3'] = df['family_size'] == 1
    df['ticket_class_1'] = df['Ticket'].str.startswith('CA') | df['Ticket'].str.startswith('C.A')
    df['ticket_class_2'] = df['Ticket'].str.len() == 4 & (df['Ticket'].str.startswith('1')
                                                          | df['Ticket'].str.startswith('2')| df['Ticket'].str.startswith('7'))
    df['ticket_class_3'] = df['Ticket'].str.len() == 6 & (df['Ticket'].str.startswith('3'))
    df['ticket_class_4'] = df['Ticket'].str.startswith('S.O.C')
    df['ticket_class_5'] = df['Ticket'].str.startswith('PC')
    df['ticket_class_6'] = df['Ticket'].str.startswith('SC')
    df['ticket_class_7'] = df['Ticket'].str.startswith('W./C.')
    df['ticket_class_8'] = df['Ticket'].str.len() == 5 & (df['Ticket'].str.startswith('1') | df['Ticket'].str.startswith('2'))
    df['is_mr'] = df['Name'].str.contains('Mr')
    df['is_mrs'] = df['Name'].str.contains('Mrs') | df['Name'].str.contains('Dona')
    df['is_miss'] = df['Name'].str.contains('Miss')
    df['is_mr'] = df['Name'].str.contains('Mr')
    df['is_master'] = df['Name'].str.contains('Master')
    df['is_dr'] = df['Name'].str.contains('Dr')
    df['is_rev'] = df['Name'].str.contains('Rev')
    df['name_len'] = df['Name'].str.len()
    df['noname'] = ~(df['is_mr'] & df['is_mrs'] & df['is_miss'] & df['is_mr'] & df['is_master']
                     & df['is_dr']& df['is_rev'])
    for clazz in df['Pclass'].unique():
        df['is_%d_class' % clazz] = df['Pclass'] == clazz
    return df


train, test = get_data('data/train.csv', 'data/test.csv')
X, y = extract_target_column(train, 'Survived')

cv = KFold(n_splits=3, shuffle=True)
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
clf = XGBClassifier()
acc = cross_val_score(clf, X, y, scoring='accuracy', cv=cv)
print(min(acc), mean(acc))
