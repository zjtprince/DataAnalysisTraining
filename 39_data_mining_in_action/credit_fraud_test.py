
# -*- coding:utf-8 -*-
# 使用逻辑回归对信用卡欺诈进行分类
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR,LinearSVC
import warnings
warnings.filterwarnings('ignore')

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap = plt.cm.Blues):

    plt.figure()
    plt.imshow(cm, cmap=cmap, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=0)
    plt.yticks(tick_marks,classes)

    thresh = cm.max()/2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment = 'center', color = 'white' if cm[i,j ] > thresh else 'black')

    plt.ylabel("True label")
    plt.xlabel("predict label")
    plt.show()

def show_metrics(cm):
    tp = cm[1,1]
    fn = cm[1,0]
    fp = cm[0,1]
    tn = cm[0,0]
    # https://www.jianshu.com/p/4434ea11c16c
    p = tp/(tp+fp)
    print("precision:{:.3f}".format(p))
    r= tp /(tp + fn)
    print("recall:{:.3f}".format(r))
    print("f1:{:.3f}".format(2*(p*r/(p+r))))


def plot_precision_recall():
    plt.step(recall, precision, color = 'b', alpha = 0.2, where = 'post')
    plt.fill_between(recall, precision, step ='post', alpha = 0.2, color = 'b')
    plt.plot(recall, precision, linewidth=2)
    plt.xlim([0.0,1])
    plt.ylim([0.0,1.05])
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title('精确率-召回率 曲线')
    plt.show();

def plot_feature_importance(clf, train_features):
    coeffs = clf.feature_importances_
    df_co = pd.DataFrame(coeffs, columns=["importance_"])
    # 下标设置为Feature Name
    df_co.index = train_features.columns
    df_co.sort_values("importance_", ascending=True, inplace=True)
    df_co.importance_.plot(kind="barh")
    plt.title("Feature Importance")
    plt.show()

data = pd.read_csv('~/Documents/creditcard.csv')

print(data.info())
# print (data.describe())

##显示交易条数

nums = len(data)
nums_fraud = len(data[data['Class']==1])
print("交易总条数：{}".format(nums))
print("诈骗总条数：{}".format(nums_fraud))
print("诈骗交易比例：{:.6f}".format(nums_fraud/nums))


plt.rcParams['font.sans-serif']=['SimHei']

plt.figure()
ax = sns.countplot(x = "Class",data=data)
plt.title("类别分部")
# plt.show()

# 欺诈和正常交易可视化
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15,8))
bins = 50
ax1.hist(data.Time[data.Class == 1], bins = bins, color = 'deeppink')
ax1.set_title('诈骗交易')
ax2.hist(data.Time[data.Class == 0], bins = bins, color = 'deepskyblue')
ax2.set_title('正常交易')
plt.xlabel('时间')
plt.ylabel('交易次数')
# plt.show()


data['Amount_Norm'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))

y = np.array(data.Class.tolist())
data = data.drop(['Time','Amount','Class'], axis=1)
X = np.array(data.values)

train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=33, test_size=0.1)

reg = LogisticRegression()
reg.fit(train_x, train_y)
pred_y = reg.predict(test_x)
cm = confusion_matrix(test_y, pred_y)
class_names = [0,1]
# plot_confusion_matrix(cm, classes=class_names, title='混淆矩阵' )
# show_metrics(cm)
#
# score_y = reg.decision_function(test_x)
# precision, recall, thresholds = precision_recall_curve(test_y, score_y)
# plot_precision_recall()

plot_feature_importance(reg, test_x)

precision:0.830
recall:0.650
f1:0.729

precision:0.848
recall:0.650
f1:0.736
