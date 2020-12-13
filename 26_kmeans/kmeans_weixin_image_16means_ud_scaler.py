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
            data.append([(c1+1)/256, (c2+1)/256, (c3+1)/256])
    f.close()
    return np.mat(data), width, height


matrix, width, height = load_data("/home/zjtprince/Documents/kmeans/weixin.jpg")

model = KMeans(n_clusters=16)
# print(width, height)
labels = model.fit_predict(matrix)
# print(labels[:5])
# print(labels.shape)
# print(type(labels))
labels = np.reshape(labels ,[width, height])
# print(labels.shape)
# pic_mark = Image.new("L", (width, height))


# 将聚类标识矩阵转化为不同颜色的矩阵
img=Image.new('RGB', (width, height))
for x in range(width):
    for y in range(height):
        c1 = model.cluster_centers_[labels[x, y], 0]
        c2 = model.cluster_centers_[labels[x, y], 1]
        c3 = model.cluster_centers_[labels[x, y], 2]
img.putpixel((x, y), (int(c1*256)-1, int(c2*256)-1, int(c3*256)-1))
img.save('weixin_new.jpg')