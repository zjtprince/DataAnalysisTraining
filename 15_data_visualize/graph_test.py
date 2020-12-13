#!/usr/bin/python3

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def scatter_figure():
    x = np.random.randn(1000)
    y = np.random.randn(1000)
    #plt.scatter(x, y ,marker='x')
    #plt.show();
    
    df = pd.DataFrame({"x":x,"y":y})
    
    sns.jointplot(x="x",y="y",data =df , kind="scatter")
    plt.show()
def line_plot_figure():
    x = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
    y = [5, 3, 6, 20, 17, 16, 19, 30, 32, 35]
    # 使用Matplotlib画折线图
    plt.plot(x, y)
    plt.show()
    # 使用Seaborn画折线图
    df = pd.DataFrame({'x': x, 'y': y})
    sns.lineplot(x="x", y="y", data=df)
    plt.show()


def distplot_figure():
    
    a = np.random.randn(100)
    s = pd.Series(a)
    # 用Matplotlib画直方图
    plt.hist(s)
    plt.show()
    # 用Seaborn画直方图
    sns.distplot(s, kde=False)
    plt.show()
    sns.distplot(s, kde=True)
    plt.show()



def bar_figure():
    # 数据准备
    x = ['Cat1', 'Cat2', 'Cat3', 'Cat4', 'Cat5']
    y = [5, 4, 8, 12, 7]
    # 用Matplotlib画条形图
    plt.bar(x, y)
    plt.show()
    # 用Seaborn画条形图
    sns.barplot(x, y)
    plt.show()


def boxplot_figure():
    # 数据准备
    # 生成10*4维度数据
    data = np.random.normal(size=(10, 4))
    labels = ['A', 'B', 'C', 'D']
    # 用Matplotlib画箱线图
    plt.boxplot(data, labels=labels)
    plt.show()
    # 用Seaborn画箱线图
    df = pd.DataFrame(data, columns=labels)
    sns.boxplot(data=df)
    plt.show()

#boxplot_figure()

def two_variable_distribute():

    tips = pd.DataFrame(pd.read_csv('/home/zjtprince/Documents/seaborn-data/tips.csv'))
    print(tips.head(2))
    sns.jointplot(x='total_bill',y='tip',kind='scatter',data=tips)
    sns.jointplot(x='total_bill',y='tip',kind='kde',data=tips)
    sns.jointplot(x='total_bill',y='tip',kind='hex',data=tips)
    plt.show()

def pair_plot_figure():
    # like pandas describe()
    iris = pd.DataFrame(pd.read_csv('/home/zjtprince/Documents/seaborn-data/iris.csv'))
    print(iris.head(2))

    sns.pairplot(iris)
    plt.show()

#two_variable_distribute()
#pair_plot_figure()

def car_crashes_figure():
    car_crashes = pd.DataFrame(pd.read_csv('/home/zjtprince/Documents/seaborn-data/car_crashes.csv'))
    print(car_crashes.head(2))
    sns.pairplot(car_crashes,kind='kde')
    plt.show()
#car_crashes_figure()

a = np.array([[4,3,2],[2,1,4]])
b = np.sort(a,axis=0)
c = np.sort(b)
print(a)
print(b)
print(c)
