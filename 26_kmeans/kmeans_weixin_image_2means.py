import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


def load_data(filename):
    global f
    f = open(filename, 'rb')
    image = Image.open(f)
    width, height = image.size
    data = []
    for x in range(width):
        for y in range(height):
            c1, c2, c3 = image.getpixel((x, y))
            data.append([c1, c2, c3])
    f.close()
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return np.mat(data), width, height


matrix, width, height = load_data("/home/zjtprince/Documents/kmeans/weixin.jpg")

model = KMeans(n_clusters=2)
print(width, height)
labels = model.fit_predict(matrix)
print(labels[:5])
print(labels.shape)
print(type(labels))
labels = np.reshape(labels ,[width, height])
print(labels.shape)
pic_mark = Image.new("L", (width, height))
for x in range(width):
    for y in range(height):
        #  根据类别设置图像灰度, 类别0 灰度值为255， 类别1 灰度值为127
        #labels里面为 0或者1
        pic_mark.putpixel((x, y), int(256/(labels[x][y]+1))-1)
        pic_mark.save("weixin_mark.jpg", "JPEG")