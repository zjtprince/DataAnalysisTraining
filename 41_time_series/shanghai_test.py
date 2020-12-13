import pandas as pd
from datetime import datetime
from statsmodels.tsa.arima_model import ARMA
import warnings
import matplotlib.pyplot as  plt
from itertools import product

warnings.filterwarnings('ignore')

data = pd.read_csv('~/Documents/bitcoin/shanghai_1990-12-19_to_2019-2-28.csv')
print(data.head())
print(data.info())

data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data.index = data['Timestamp']
print(data.head())

data_m = data.resample('M').mean()

q_range = range(0, 5)
p_range = range(0, 5)

params = product(q_range, p_range)
param_list = list(params)
# print(param_list)
results = []
best_aic = float('inf')

for param in param_list:
    try:
        model = ARMA(data_m['Price'], order=(param[0], param[1])).fit()
    except ValueError:
        print("参数错误", param)
        continue
    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
    results.append([param, model.aic])

result_table = pd.DataFrame(results)
result_table.columns = ['parameters', 'aic']

print("best model:", best_model.summary())
print("best param:", best_param)

data_month2 = data_m[['Price']]

date_list = [datetime(2019, 3, 31), datetime(2019, 4, 30), datetime(2019, 5, 31), datetime(2019, 6, 30),
             datetime(2019, 7, 31),
             datetime(2019, 8, 31), datetime(2019, 9, 30), datetime(2019, 10, 31), datetime(2019, 11, 30),
             datetime(2019, 12, 31)]
future = pd.DataFrame(index=date_list, columns=data_m.columns)

data_month2 = pd.concat([data_month2, future])
data_month2['forecast'] = best_model.predict(start=0, end=384)
#start=0表示从第0个数据开始计算
#end=348是指需要计算348个数据，即从1990-12-19到2019-12-31一共有348个月，所以有348个数据
print(data_month2.head())

plt.figure(figsize=(20, 7))
plt.rcParams['font.sans-serif'] = ['SimHei']
data_month2['Price'].plot(label='实际指数')
data_month2['forecast'].plot(label='预测指数', ls='--', color='r')
plt.legend()
plt.title('沪市指数（月）')
plt.xlabel('时间')
plt.ylabel('指数')
plt.show()
