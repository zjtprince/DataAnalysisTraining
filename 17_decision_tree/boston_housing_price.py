

import numpy as np
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor,export_graphviz
import graphviz

from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split

boston = load_boston()

print(boston.feature_names)

features = boston.data
prices = boston.target
print(np.alen(features), np.alen(prices))

train_features , test_features, train_prices, test_prices = train_test_split(features,prices,test_size=0.33)

print(np.alen(train_features), np.alen(test_features))

dtr = DecisionTreeRegressor()
dtr.fit(train_features, train_prices)

predict_prices = dtr.predict(test_features)


std = mean_squared_error(test_prices,predict_prices)
mean_abs_err = mean_absolute_error(test_prices,predict_prices)
print("二乘偏差均值：%f ",std)
print("绝对值偏差均值：%f ",mean_abs_err)

data =export_graphviz(dtr,out_file=None, feature_names=boston.feature_names)
graph = graphviz.Source(data)

graph.render(filename='boson_housing_prices',view=True)



