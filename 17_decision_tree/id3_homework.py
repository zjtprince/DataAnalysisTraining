
from sklearn import tree
import numpy as np
import graphviz
import os

'''
H(D) = -2 / 4 * log(2 / 4) - 2 / 4 * log(2 / 4) = 1
a: 属性“红”
a1: 属性“红”=是，H(a1) = -2 / 2 * log(2 / 2) - 0 / 2 * log(0 / 2) = 0
a2: 属性“红”=否，H(a2) = -0 / 2 * log(0 / 2) - 2 / 2 * log(2 / 2) = 0
b: 属性“大”
b1: 属性“大”=是，H(b1) = -1 / 2 * log(1 / 2) - 1 / 2 * log(1 / 2) = 1
b2: 属性“大”=否，H(b2) = -1 / 2 * log(1 / 2) - 1 / 2 * log(1 / 2) = 1

Gain(D, a) = H(D) - (2 / 4 * H(a1) + 2 / 4 * H(a2)) = 1
Gain(D, b) = H(D) - (2 / 4 * H(b1) + 2 / 4 * H(b2)) = 0

大的信息增益为0 ，决策树为：红作为最优特征，红的就是好苹果，不红的就是坏苹果
     红
是       否
好苹果    坏苹果
'''
#os.environ['PATH'] += os.pathsep + '/home/zjtprince/.local/lib/python3.8/site-packages/graphviz'


#percondition : sudo apt install graphviz
#               pip3 install graphviz

#创建数据 ['红'，‘大’] 是=1,否=0
data = np.array([[1,1],[1,0],[0,1],[0,0]])
target = np.array([1,1,0,0])
clt = tree.DecisionTreeClassifier(criterion='entropy')
clt.fit(data, target)

dot_data = tree.export_graphviz(clt, out_file=None,class_names=['好苹果','坏苹果'],feature_names=['红','大'],)
graph = graphviz.Source(dot_data)
graph.render(filename='apple-classifier',view=True)






