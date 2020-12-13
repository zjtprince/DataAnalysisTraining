import pandas as pd

from pandas import DataFrame



df1 = DataFrame({'name':['ZhangFei', 'GuanYu', 'a', 'b', 'c'], 'data1':range(5)})
df2 = DataFrame({'name':['ZhangFei', 'GuanYu', 'A', 'B', 'C'], 'data2':range(5)})

print(df1)
print(df2)

#left right inner outer
df3 = pd.merge(df1,df2,how="right",on="name")
print(df3)