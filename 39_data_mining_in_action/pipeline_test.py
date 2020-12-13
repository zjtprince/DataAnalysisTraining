
from sklearn.pipeline import  Pipeline


from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_iris

from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
parameters = {"classifier__n_estimators":range(1,11)}

rf = RandomForestClassifier()
pipeline = Pipeline([
    ('scaler',StandardScaler()),
    ('classifier',rf)
])

model = GridSearchCV(estimator=pipeline, param_grid=parameters)
model.fit(iris.data, iris.target)

print(model.best_score_, model.best_params_)