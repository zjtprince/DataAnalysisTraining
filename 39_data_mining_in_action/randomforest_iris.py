
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

iris = load_iris()
rf = RandomForestClassifier()
parameters = {'n_estimators':range(1,11)}

gridSearch = GridSearchCV(estimator=rf,param_grid=parameters)

gridSearch.fit(iris.data, iris.target)

print("best score:", gridSearch.best_score_)
print('best param:', gridSearch.best_params_)