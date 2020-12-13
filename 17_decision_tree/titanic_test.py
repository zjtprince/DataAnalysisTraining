
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import feature_extraction
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import graphviz
train_data = pd.DataFrame(pd.read_csv('~/Documents/titanic_data/train.csv'))

print(train_data.shape)

test_data = pd.DataFrame(pd.read_csv('~/Documents/titanic_data/test.csv'))
print(train_data.shape)
print('-'*30)

print(train_data.head(5))
print('-'*30)
print(train_data.tail(5))
print(train_data.info())
print('-'*30)
print(train_data.describe())
print('-'*30)
print(train_data.describe(include=['O']))


def data_cleaning():
    global train_features, train_labels, new_features,test_features
    train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
    test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)
    test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)
    # print(train_data['Embarked'].value_counts())
    train_data['Embarked'].fillna('S', inplace=True)
    test_data['Embarked'].fillna('S', inplace=True)
    print(train_data.columns)
    print(train_data.info())
    print(train_data.describe(include='O'))
    features = ['Pclass', 'Sex', "Age", 'SibSp', 'Parch', 'Fare', 'Embarked']
    train_features = train_data[features]
    train_labels = train_data['Survived']
    test_features = test_data[features]
    dict_vec = feature_extraction.DictVectorizer(sparse=False)
    train_features = dict_vec.fit_transform(train_features.to_dict(orient='record'))
    print(dict_vec.feature_names_)
    new_features = dict_vec.feature_names_


    test_features = dict_vec.fit_transform(test_features.to_dict(orient='record'))


data_cleaning()

dtc = tree.DecisionTreeClassifier()
dtc.fit(train_features,train_labels)

predict_labels = dtc.predict(test_features)
print(predict_labels[0:5])
print("score %f ",dtc.score(train_features, train_labels))

# k 折交叉验证
print("k 折交叉验证准确率：%f",np.mean(cross_val_score(dtc, train_features,train_labels,cv= 10)))

graph_data = tree.export_graphviz(dtc,out_file=None,feature_names=new_features)

graph = graphviz.Source(graph_data)
graph.view(filename='titanic_classifier',quiet_view=False)

