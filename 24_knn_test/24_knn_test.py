
from sklearn.neighbors import KNeighborsClassifier

from sklearn.datasets import  load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import  accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

digits = load_digits()

data = digits.data

# print(data.shape)
# print(digits.images[0])
# print(digits.target[0])

plt.gray()
plt.imshow(digits.images[0])
#plt.show()


#train_x, train_y, test_x, test_y = train_test_split(data, digits.target, test_size=0.25, random_state=33)
train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size=0.25, random_state=33)

ss = StandardScaler()
# ss = MinMaxScaler() # for MultinomialNB
train_x = ss.fit_transform(train_x)
test_x = ss.transform(test_x)




model = KNeighborsClassifier()  #97%
# model = KNeighborsClassifier(n_neighbors=200)  #84%
# model = SVC()  #98%

# model = MultinomialNB(alpha=0.001) # 89
# model = DecisionTreeClassifier()  #85

model.fit(train_x,train_y)
prediction = model.predict(test_x)
print("accuracy : %f"  %accuracy_score(test_y,prediction))

