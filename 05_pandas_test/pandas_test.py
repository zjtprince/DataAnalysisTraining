#!/usr/bin/python3
#vim: set fileencoding:utf-8

import pandas as pd

from pandas import DataFrame, Series

#
# a = Series([1,2,3,4])
# b = Series([1,2,3,4],index={'a','b','c','d'})
# c = Series([1,2,3,4],index=('a','b','c','d'))
# d = Series([1,2,3,4],index=['a','b','c','d'])
#
# print(a)
# print(b)
# print(c)
# print(d)
#
#
# d = {'a':1,"b":2,"c":3,"d":4}
#
# a1 = Series(d)
# print(a1)


data = {'Chinese': [66, 95, 93, 90,80],'English': [65, 85, 92, 88, 90],'Math': [30, 98, 96, 77, 90]}

d1 = DataFrame(data)
print(d1)

index = ['zhangfei','zhangyun','guangyu','huangzong','dianwei']
df = DataFrame(data=data,index=index)

# read from excel file
#data = pd.read_excel('/home/zjtprince/sanguo_heros_score.xlsx',header=0,index_col=0)
##print(data)
#df = DataFrame(data,dtype="int64")
#print(df)
#print(df.dtypes)


#print(df.drop(columns=["Chinese"]))
#print(df.drop(index="zhangfei"))
#print(df.rename(columns={"Chinese":'yuwen',"English":"yingyu"},inplace=True))
#print(df)
#df = df.drop_duplicates()
#print(df)
#
#print(df.isnull().any())
#
df['name'] = df.index
print(df)

df['name'] = df['name'].str.upper()
df.columns= df.columns.str.upper()
df.columns= df.columns.str.title()
print(df)

df["语文"] = df['Chinese'].apply(lambda x: 2*x)
print(df)
print(df.describe())