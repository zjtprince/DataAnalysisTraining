import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from skimage import color

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

model = KMeans(n_clusters=16)
print(width, height)
labels = model.fit_predict(matrix)
print(labels[:5])
print(labels.shape)
print(type(labels))
labels = np.reshape(labels ,[width, height])
print(labels.shape)
pic_mark = Image.new("L", (width, height))




# 将聚类标识矩阵转化为不同颜色的矩阵
label_color = (color.label2rgb(labels)*255).astype(np.uint8)
label_color = label_color.transpose(1,0,2)
images = Image.fromarray(label_color)
images.save('weixin_mark_color.jpg')