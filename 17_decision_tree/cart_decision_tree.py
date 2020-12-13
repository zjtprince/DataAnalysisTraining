
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.datasets import load_iris
import numpy as np
import graphviz
iris = load_iris()

features = iris.data
labels = iris.target

#print(iris)

#随机抽取33 % 的数据作为测试集，其余为训练集
train_features, test_features, train_labels, test_labels = train_test_split(features, labels,test_size=0.33,random_state=0)
# 创建CART分类树
clf = DecisionTreeClassifier(criterion='gini')
# 拟合构造CART分类树
clf = clf.fit(train_features, train_labels)
# 用CART分类树做预测
test_predict = clf.predict(test_features)
# 预测结果与测试集结果作比对
score = accuracy_score(test_labels, test_predict)

print("CART分类树准确率 %.4lf" % score)

dot_data = export_graphviz(clf, out_file=None,feature_names=['sepal_length','sepal_width','petal_length','petal_width']
                           ,class_names=['setosa', 'versicolor', 'virginica'])
graph = graphviz.Source(dot_data)
graph.render(filename='iris-classifier',view=True)
