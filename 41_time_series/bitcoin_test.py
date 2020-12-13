import pandas as pd
from datetime import datetime
from statsmodels.tsa.arima_model import ARMA
import  warnings
import matplotlib.pyplot as  plt
from itertools import product
warnings.filterwarnings('ignore')



data = pd.read_csv('~/Documents/bitcoin/bitcoin_2012-01-01_to_2018-10-31.csv')
print(data.head())
# print(data.info())


data['Timestamp'] = pd.to_datetime(data['Timestamp'] )
data.index = data['Timestamp']
print(data.head())

data_m = data.resample('M').mean()
data_q = data.resample('Q-DEC').mean()
data_y = data.resample('A-DEC').mean()


fig = plt.figure(figsize=(15,7))
plt.rcParams['font.sans-serif']=['SimHei']
plt.suptitle('比特币金额（美元）',fontsize=20   )

plt.subplot(221)
plt.plot(data['Weighted_Price'],'-',label='按天')
plt.legend()
plt.subplot(222)
plt.plot(data_m['Weighted_Price'],'-',label='按月')
plt.legend()
plt.subplot(223)
plt.plot(data_q['Weighted_Price'],'-',label='按季')
plt.legend()
plt.subplot(224)
plt.plot(data_y['Weighted_Price'],'-',label='按年')
plt.legend()

# plt.show()

q_range = range(0,5)
p_range = range(0,5)

params = product(q_range,p_range)
param_list = list(params)
print(param_list)

results = []

best_aic = float('inf')


for param in param_list:
    try:
        model = ARMA(data_m['Weighted_Price'],order=(param[0],param[1])).fit()

    except ValueError:
        print("参数错误",param)
        continue
    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
    results.append([param,model.aic])

result_table = pd.DataFrame(results)
result_table.columns=['parameters','aic']

print("best model:", best_model.summary())
print("best param:", best_param)


data_month2 = data_m[['Weighted_Price']]

date_list = [datetime(2018, 11, 30), datetime(2018, 12, 31), datetime(2019, 1, 31), datetime(2019, 2, 28), datetime(2019, 3, 31),
             datetime(2019, 4, 30), datetime(2019, 5, 31), datetime(2019, 6, 30)]
future = pd.DataFrame(index=date_list, columns=data_m.columns)

data_month2 = pd.concat([data_month2,future])
data_month2['forecast']= best_model.predict(start=0, end=91)
print(data_month2.head())

plt.figure(figsize=(20 ,7))

data_month2['Weighted_Price'].plot(label='实际金额')
data_month2['forecast'].plot(label='预测金额',ls='--',color='r')
plt.legend()
plt.title('比特币金额（月）')
plt.xlabel='时间'
plt.ylabel='金额'
plt.show()