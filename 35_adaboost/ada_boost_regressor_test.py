
from sklearn.ensemble import AdaBoostClassifier,AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


#base_estimator 弱分类器，默认是决策树
#n_estimators   算法迭代次数，也就是分类器个数
abc = AdaBoostClassifier(base_estimator=None, n_estimators=50,learning_rate=1, algorithm='SAMME.R',random_state=None)
abr = AdaBoostRegressor(base_estimator=None, n_estimators=50,learning_rate=1, loss='linear',random_state=None)

data = load_boston()

train_x, test_x , train_y , test_y = train_test_split(data.data, data.target, test_size=0.3, random_state=33)

abr.fit(train_x,train_y)
pred_y = abr.predict(test_x)
print("房价预测结果：", pred_y)
print("均方差：", mean_squared_error(test_y, pred_y))

for i in zip(test_y, pred_y):
    print(i)






