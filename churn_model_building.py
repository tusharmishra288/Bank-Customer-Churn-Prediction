#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SVMSMOTE
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import RepeatedStratifiedKFold,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.metrics import classification_report,roc_auc_score,plot_confusion_matrix,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from xgboost import XGBClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import joblib

#loading of training data
train=pd.read_csv('churn_train.csv')
print(train.shape)

#loading of test data
test=pd.read_csv('churn_test.csv')
print(test.shape)



#removing unecessary features
train.drop(columns=['Unnamed: 0','branch_code','city','customer_id'],inplace=True)
test.drop(columns=['Unnamed: 0','branch_code','city','customer_id'],inplace=True)


#encoding of categorical variables in training dataset
train['gender'].replace({'Male':0,'Female':1},inplace=True)
train['occupation'].replace({'self_employed':1,'salaried':2,'student':3,'retired':4,'company':5},inplace=True)

#encoding of categorical variables in testing dataset
test['gender'].replace({'Male':0,'Female':1},inplace=True)
test['occupation'].replace({'self_employed':1,'salaried':2,'student':3,'retired':4,'company':5},inplace=True)


#preparing training and test datasets
X_train=train.drop('churn',axis=1)
y_train=train.churn

X_test=test.drop('churn',axis=1)
y_test=test.churn



print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=25)

"""
# get a list of models to evaluate
def get_models():
	models = dict()
	for i in range(2, 18):
		rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=i)
		model = DecisionTreeClassifier()
		models[str(i)] = Pipeline(steps=[('s',rfe),('m',model)])
	return models
 
# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
	scores = cross_val_score(model, X, y, scoring='f1', cv=cv, n_jobs=-1, error_score='raise')
	return scores
 
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model, X_train, y_train)
	results.append(scores)
	names.append(name)
	print('>%s %.3f' % (name, np.mean(scores)))
"""

#selecting best features found above 
model = DecisionTreeClassifier(random_state=25)
rfe = RFE(model, n_features_to_select=7)
fit = rfe.fit(X_train, y_train)
result=X_train.columns[fit.support_]
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (result))


X_train=X_train[result]
X_test=X_test[result]



#performing over sampling using SVM SMOTE and undersamping using Random Under Sampler
over = SVMSMOTE(random_state=25)
under= RandomUnderSampler(random_state=25)

steps=[('o',over),('u',under)]
pipe=Pipeline(steps=steps)
X_train_ns,y_train_ns=pipe.fit_resample(X_train,y_train)

#scaling of data using robustscaler
rs=RobustScaler()
X_train_scaled=rs.fit_transform(X_train_ns)
X_test_scaled=rs.transform(X_test)

#function to fit baseline models in training set with K Fold cross validation
def model(mod):
    scores = cross_val_score(mod, X_train_scaled, y_train_ns, scoring='f1', cv=cv, n_jobs=-1)
    return np.mean(scores)
    

lr=model(LogisticRegression())

dt=model(DecisionTreeClassifier())

xgb=model(XGBClassifier(label_encoder=False))

gbm=model(GradientBoostingClassifier())

rf=model(RandomForestClassifier())

#since random forest performed well therefore tuning it using hyperopt
params = {
    'max_depth': hp.quniform('max_depth',1,20,1),
    'n_estimators': hp.choice('n_estimators',range(1,200))
}

#objective function
def tune_model(params): 
     classifier=RandomForestClassifier(**params)
     classifier.fit(X_train_scaled,y_train_ns)
     mod_pred=classifier.predict(X_test_scaled)
     f1=f1_score(y_test,mod_pred)
     return {'loss': -f1,'status':STATUS_OK}

#trials
trials = Trials()
best = fmin(fn=tune_model,
            space=params,
            algo=tpe.suggest,
            max_evals=300,
            trials=trials)

print(best)

#best hyperparameters
#{'max_depth': 18.0,'n_estimators': 102}

#retraining the model using best hyperparameters
model=RandomForestClassifier(max_depth=18.0,n_estimators=102)

#predicting values from tuned model
def predict_values(X,y,model):
    model.fit(X_train_scaled,y_train_ns)
    mod_pred=model.predict(X)
    print(classification_report(y,mod_pred))
    print(roc_auc_score(y,mod_pred))
    plot_confusion_matrix(model,X,y,cmap=plt.cm.Blues)

#predicting on test dataset
predict_values(X_test_scaled,y_test,model)


# save the model to disk using joblib
filename = 'finalized_model.sav'
joblib.dump(model,filename)
 
