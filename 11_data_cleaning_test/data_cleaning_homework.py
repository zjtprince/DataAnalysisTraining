import pandas as pd
from pandas import DataFrame
csvfile = pd.read_csv('./cloth_shop.csv',header=None)

df = DataFrame(csvfile)


df.columns =['Name','Age','Weight','m0006','m0612','m1218','f0006','f0612','f1218']

##删除全空的行
df.dropna(inplace=True,how='all')

##最高频值填充
age_maxf = df['Age'].value_counts().index[0]
df['Age'].fillna(age_maxf,inplace=True)

#拆分name
df[['First_Name','Last_Name']] = df['Name'].str.split(expand=True)
#df.insert(0,['First_Name','Last_Name'],df['Name'].str.split(expand=True))
df.drop('Name', axis=1, inplace=True)
# 删除非 ASCII 字符
df[['First_Name','Last_Name']].replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)

df.drop_duplicates(['First_Name','Last_Name'], inplace=True)



# 获取 weight 数据列中单位为 lbs 的数据
rows_with_lbs = df['Weight'].str.contains('lbs').fillna(False)
print ( df[rows_with_lbs])
# 将 lbs转换为 kgs, 2.2lbs=1kgs
for i,lbs_row in df[rows_with_lbs].iterrows():
  # 截取从头开始到倒数第三个字符之前，即去掉lbs。
  weight = int(float(lbs_row['Weight'][:-3])/2.2)
  df.at[i,'Weight'] = '{}kgs'.format(weight)

print(df)