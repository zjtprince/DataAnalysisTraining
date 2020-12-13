

import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.cluster import KMeans

data  = pd.read_csv('~/Documents/kmeans/data.csv',encoding='gbk')

print(data.info())
print(data.head())

features = data.columns[1:]

train_x = data[features]

# scaler = MinMaxScaler()
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)

model = KMeans(n_clusters=5)

clusters = model.fit_predict(train_x)
data['聚类'] = clusters
print(data)

