import numpy as np

a = np.array([1, 2, 3])
print(a, type(a),a.shape, a.dtype)

print ()

b = np.array([[1,2,3],[4,5,6],[7,8,9]])
b[1,1] = 10
print (b, type(b), b.shape,b.dtype)


persontype = np.dtype({
    'names':['name','age','chinese','math','english'],
    'formats':['S32','i','i','i','f']
})

persons = np.array([("zhangfei",32,75,80,78),("guanyu",36,60,50,40),("liubei",40,50,67,92),("zhaoyun",20,52,75,59)],
                   dtype=persontype)

ages = persons[:]["age"]
chineses = persons[:]['chinese']
math_score = persons[:]['math']
english_score = persons[:]['english']

print(np.mean(ages),chineses, math_score, english_score)

print(list(zip(ages, chineses,math_score,english_score)))

x1 = np.arange(1,10,2)
x2 = np.linspace(1,9,5)
print(x1, x1.dtype ,x2,x2.dtype)

print(np.add(x1,x2))
print(np.subtract(x1, x2))
print(np.multiply(x1,x2))
print(np.power(x1,x2))
print(np.remainder(x1,x2))

a= np.array([[1,2,3],[4,5,6],[7,8,9]])

print (np.amin(a))
print (np.amin(a,0))
print (np.amin(a,1))

print (np.amax(a))
print (np.amax(a,0))
print (np.amax(a,1))

print(np.ptp(a))
print(np.ptp(a,0))
print(np.ptp(a,1))

print(np.percentile(a,q=50))
print(np.percentile(a,axis=0,q=50))
print(np.percentile(a,axis=1,q=50))
print(np.percentile(a,q=75))
#中位数
print (np.median(a))
print (np.median(a,0))
print (np.median(a,1))
#平均值 also see average
print (np.mean(a))
print (np.mean(a,0))
print (np.mean(a,1))

# weighted average
wts = np.array([1,2,3,4])
a = np.array([1,2,3,4])
print (np.average(a))
print (np.average(a,weights=wts))
print(np.std(a))
print(np.var(a))

a= np.array([[4,3,2],[2,4,1]])

print(np.sort(a))
print(np.sort(a,axis=None))
print(np.sort(a,axis=0))
print(np.sort(a,axis=1))


