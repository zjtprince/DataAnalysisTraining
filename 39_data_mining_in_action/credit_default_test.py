
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,learning_curve,GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import  RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from  sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier

data = pd.read_csv("~/Documents/credit_default/UCI_Credit_Card.csv")

columns = data.columns.tolist()
columns.remove('default.payment.next.month')
columns.remove('ID')
# print(columns)
features = data[columns].values
target = data['default.payment.next.month'].values

train_x, test_x, train_y, test_y = train_test_split(features, target, stratify=target,random_state=1,test_size=0.3)

classifiers = [
    AdaBoostClassifier(random_state=1, learning_rate=1)
]
classifier_names = [
    'ada'
]
classifier_param_grid = [
    {'ada__n_estimators':[10,50, 100]}
]

def search(pipeline, grid_param,train_x, train_y, test_x, test_y,scoring='accuracy'):
    response= {}
    s = GridSearchCV(estimator=pipeline,param_grid=grid_param,scoring=scoring)
    s.fit(train_x,train_y)
    pred_y = s.predict(test_x)
    print("best score:",s.best_score_)
    print("best params:",s.best_params_)
    print('准确率：', accuracy_score(test_y,pred_y))
    response['predict_y'] = pred_y
    response['accuracy_score'] = accuracy_score(test_y, pred_y)
    return response

for name , classifier, param in  zip(classifier_names, classifiers, classifier_param_grid):
    pipeline = Pipeline([
        ('scaler',StandardScaler()),
        (name, classifier)
    ])
    result = search(pipeline,param,train_x,train_y,test_x,test_y)

#best score: 0.8188095238095239
#best params: {'ada__n_estimators': 10}
#准确率： 0.8128888888888889