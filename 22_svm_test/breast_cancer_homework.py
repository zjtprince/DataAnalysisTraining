import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

df = pd.read_csv('~/Documents/breast_cancer_data/data.csv')
print(df.info())

features = df.columns[2:]
#features = ['radius_mean','texture_mean', 'smoothness_mean','compactness_mean','symmetry_mean', 'fractal_dimension_mean']
features = ['radius_mean','perimeter_mean','area_mean','concave points_mean','radius_worst','perimeter_worst','area_worst','concave points_worst']
# 将B良性替换为0，M恶性替换为1
df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})

train , test = train_test_split(df, test_size=0.3)

train_x = train[features]
train_y = train['diagnosis']
test_x = test[features]
test_y = test['diagnosis']

##z_score标准化数据
ss = StandardScaler()
train_x = ss.fit_transform(train_x)
test_x = ss.transform(test_x)

model = svm.SVC()

model.fit(train_x, train_y)
prediction = model.predict(test_x)
print('accuracy : %f',accuracy_score(prediction, test_y))
print("k 折交叉验证训练集准确率：%f",np.mean(cross_val_score(model, train_x,train_y,cv=10)))
print("k 折交叉验证测试集准确率：%f",np.mean(cross_val_score(model, test_x,prediction,cv=10)))
## SVC       30个特征值  准确度为：98 %
## SVC        6个特征值  准确度为：92 %
## LinearSVC 30个特征值  准确度为：97 %
## LinearSVC  6个特征值  准确度为：92 %




