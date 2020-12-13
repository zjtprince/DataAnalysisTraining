
from sklearn.ensemble import AdaBoostClassifier,AdaBoostRegressor
from sklearn.metrics import mean_squared_error,zero_one_loss
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree  import DecisionTreeClassifier
from  sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np


n_estimators = 200
x, y = datasets.make_hastie_10_2(12000,random_state=11)

train_x = x[2000:]
train_y = y[2000:]

test_x =  x[:2000]
test_y = y[:2000]


#base_estimator 弱分类器，默认是决策树
#n_estimators   算法迭代次数，也就是分类器个数

#弱分类器
dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
dt_stump.fit(train_x,train_y)
dt_stump_err = 1 - dt_stump.score(test_x,test_y)


dtc = DecisionTreeClassifier()
dtc.fit(train_x,train_y)
dtc_err = 1 - dtc.score(test_x,test_y)

ada = AdaBoostClassifier(base_estimator=dt_stump, n_estimators=n_estimators,learning_rate=1, algorithm='SAMME.R',random_state=None)
ada.fit(train_x,train_y)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot([1,n_estimators],[dt_stump_err]*2, 'k-', label=u'weak decision tree model error ')
ax.plot([1,n_estimators],[dtc_err]*2,'k--', label=u'normal decision tree model error')
ada_err = np.zeros((n_estimators,))

print("ada error ",1 - ada.score(test_x,test_y))


# 遍历每次迭代的结果 i为迭代次数, pred_y为预测结果
for i,pred_y in enumerate(ada.staged_predict(test_x)):
    # 统计错误率
    ada_err[i]=zero_one_loss(test_y, pred_y)
# 绘制每次迭代的AdaBoost错误率
ax.plot(np.arange(n_estimators)+1, ada_err, label='AdaBoost Classifier Error', color='orange')
ax.set_xlabel('iteration times ')
ax.set_ylabel('error')
leg=ax.legend(loc='upper right',fancybox=True)
# plt.savefig('dtc')
plt.show()







