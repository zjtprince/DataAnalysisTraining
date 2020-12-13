import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd

df = pd.read_csv('~/Documents/breast_cancer_data/data.csv')
#pd.set_option('display.max_columns',None)
# print(df.shape)
#
# print(df.head())
# print(df.columns)
print(df.info())
# print(df.describe())
#
# index = list(range(2,32))
# print(index)
# sub_df = df.iloc[ :,index]

features_mean = df.columns[2:12]
features_se = df.columns[12:22]
features_worst = df.columns[22:32]

## df['id'] 得到的是Series , 所以 axis=1
df.drop('id',inplace=True,axis=1)

df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})

# sns.countplot(df['diagnosis'])
# plt.show()
#
# corr = df[features_mean].corr()
# plt.figure(figsize=(14,14))
# sns.heatmap(corr,annot=True)
# plt.show()
# radius_mean、perimeter_mean、area_mean 这三个属性相关性大，
# compactness_mean、daconcavity_mean、concave points_mean 这三个属性相关性大。
# 我们分别从这 2 类中选择 1 个属性作为代表，比如 radius_mean 和 compactness_mean

# 特征选择
features_remain = ['radius_mean','texture_mean', 'smoothness_mean','compactness_mean','symmetry_mean', 'fractal_dimension_mean']

train , test  = train_test_split(df, test_size=0.3)

train_x = train[features_remain]
train_y = train['diagnosis']
test_x = test[features_remain]
test_y = test['diagnosis']


##z_score标准化数据
ss = StandardScaler()

train_x = ss.fit_transform(train_x)
test_x = ss.fit_transform(test_x)

model = svm.SVC()

model.fit(train_x, train_y)

prediction = model.predict(test_x)

print('accuracy : %f',accuracy_score(prediction, test_y))

