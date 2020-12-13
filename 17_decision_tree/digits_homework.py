

import numpy as np
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeRegressor,export_graphviz,DecisionTreeClassifier
import graphviz

from sklearn.metrics import mean_squared_error,mean_absolute_error,accuracy_score
from sklearn.model_selection import train_test_split

boston = load_digits()

print(boston.feature_names)

features = boston.data
prices = boston.target
print(np.alen(features), np.alen(prices))

train_features , test_features, train_prices, test_prices = train_test_split(features,prices,test_size=0.25)

print(np.alen(train_features), np.alen(test_features))

# dtr = DecisionTreeClassifier(criterion='gini')
# dtr = DecisionTreeClassifier(criterion='entropy')
dtr = DecisionTreeClassifier()
dtr.fit(train_features, train_prices)

predict_prices = dtr.predict(test_features)


accur = accuracy_score(test_prices,predict_prices)

print('准确度为%f',accur)

data =export_graphviz(dtr,out_file=None, feature_names=boston.feature_names)
graph = graphviz.Source(data)

graph.render(filename='digits',view=False)



