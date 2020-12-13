
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import feature_extraction
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import  AdaBoostClassifier

# load dataset
train_data = pd.DataFrame(pd.read_csv('~/Documents/titanic_data/train.csv'))
test_data = pd.DataFrame(pd.read_csv('~/Documents/titanic_data/test.csv'))

# data cleaning
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)
train_data['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S', inplace=True)
# select features
features = ['Pclass', 'Sex', "Age", 'SibSp', 'Parch', 'Fare', 'Embarked']
train_x = train_data[features]
train_y = train_data['Survived']
test_x = test_data[features]
# one-hot
dict_vec = feature_extraction.DictVectorizer(sparse=False)
train_x = dict_vec.fit_transform(train_x.to_dict(orient='record'))
test_x = dict_vec.transform(test_x.to_dict(orient='record'))
print(dict_vec.feature_names_)

# decision tree
dtc = tree.DecisionTreeClassifier()
dtc.fit(train_x, train_y)

print("决策树准确率", dtc.score(train_x, train_y))
print("决策树：k折交叉验证准确率：", np.mean(cross_val_score(dtc, train_x, train_y, cv= 10)))

# adaboost
ada = AdaBoostClassifier(n_estimators=50)
ada.fit(train_x, train_y)
print("AdaBoost准确率", ada.score(train_x, train_y))
print("AdaBoost k折交叉验证准确率：", np.mean(cross_val_score(ada, train_x, train_y, cv= 10)))
