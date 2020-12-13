
import pandas as pd

from pandas import DataFrame


df = DataFrame(pd.read_excel('~/pandas_homework_score.xlsx'))


print(df.describe())

df = df.drop_duplicates()
df.fillna({'英语':df['英语'].mean(),'语文':0},inplace=True)

#df['总分'] = df['语文']+df['数学']+df['英语']
df['总分'] = df.sum(axis = 1)

print(df)