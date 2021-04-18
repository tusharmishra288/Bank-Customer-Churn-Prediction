#importing necessary libraries
import tensorflow as tf
import tensorflow.keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SVMSMOTE
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.metrics import classification_report,roc_auc_score
import seaborn as sn
import kerastuner
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from kerastuner.tuners import RandomSearch

#loading of training data
train=pd.read_csv('churn_train.csv')
print(train.shape)

validation=pd.read_csv('churn_validation.csv')
print(validation.shape)

#loading of test data
test=pd.read_csv('churn_test.csv')
print(test.shape)



#removing unecessary features
train.drop(columns=['Unnamed: 0','branch_code','city','customer_id'],inplace=True)
validation.drop(columns=['Unnamed: 0','branch_code','city','customer_id'],inplace=True)
test.drop(columns=['Unnamed: 0','branch_code','city','customer_id'],inplace=True)




#encoding of categorical variables in training dataset
train['gender'].replace({'Male':0,'Female':1},inplace=True)
train['occupation'].replace({'self_employed':1,'salaried':2,'student':3,'retired':4,'company':5},inplace=True)

#encoding of categorical variables in validation dataset
validation['gender'].replace({'Male':0,'Female':1},inplace=True)
validation['occupation'].replace({'self_employed':1,'salaried':2,'student':3,'retired':4,'company':5},inplace=True)

#encoding of categorical variables in testing dataset
test['gender'].replace({'Male':0,'Female':1},inplace=True)
test['occupation'].replace({'self_employed':1,'salaried':2,'student':3,'retired':4,'company':5},inplace=True)



X_train=train.drop('churn',axis=1)
y_train=train.churn


X_valid=validation.drop('churn',axis=1)
y_valid=validation.churn

X_test=test.drop('churn',axis=1)
y_test=test.churn



print(X_train.shape,y_train.shape)
print(X_valid.shape,y_valid.shape)
print(X_test.shape,y_test.shape)

"""
#to determine how many number of features are needed for effective modelling
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
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
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
rfe = RFE(model, n_features_to_select=6)
fit = rfe.fit(X_train, y_train)
result=X_train.columns[fit.support_]
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (result))




X_train=X_train[result]
X_valid=X_valid[result]
X_test=X_test[result]



#performing over sampling using SVM SMOTE
over = SVMSMOTE(random_state=25)
under= RandomUnderSampler(random_state=25)



#applying separately 
steps=[('o',over),('u',under)]
pipeline=Pipeline(steps=steps)
X_train_ns, y_train_ns = pipeline.fit_resample(X_train, y_train)



y_train_ns.value_counts()


from sklearn.preprocessing import RobustScaler
rs=RobustScaler()
X_train_ns=rs.fit_transform(X_train_ns)
X_valid=rs.transform(X_valid)
X_test=rs.transform(X_test)


callback=tf.keras.callbacks.EarlyStopping(monitor='recall',patience=2)

def building_model(hp):
    model = Sequential()
    for i in range(hp.Int('num_layers', 2, 20)):
            model.add(Dense(units=hp.Int('units_',min_value=32,max_value=512,
                                         step=32,default=50),activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice(
    'learning_rate',values=[0.01,0.1,0.05,0.0005,0.0001,0.005,0.001],
    default=0.1)),
    loss='binary_crossentropy'
    ,metrics=[])
    return model

tuner = RandomSearch(
    building_model,
    objective =kerastuner.Objective('val_recall',direction='max'),
    max_trials=30,
    metrics=[tf.keras.metrics.Precision(name='precision'),
              tf.keras.metrics.Recall(name='recall')],
    executions_per_trial=1,project_name='tuner')

tuner.search(X_train_ns,
             y_train_ns,
             verbose=2, 
             epochs=5,
            callbacks=[callback],validation_data=(X_valid,y_valid)) 

tuner.get_best_hyperparameters()[0].values

#{'num_layers': 3, 'units_': 372}

tuner.results_summary()

# Retrieve the best model.
best_model = tuner.get_best_models(num_models=1)[0]


best_model.evaluate(X_test, y_test)



mod_pred=best_model.predict(X_test)
mod_pred


y_pred= best_model.predict_classes(X_test)
y_pred



print(classification_report(y_test,y_pred))
print(roc_auc_score(y_test,y_pred))




cm = tf.math.confusion_matrix(labels=y_test,predictions=y_pred)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d',cmap=plt.cm.Blues)
plt.xlabel('Predicted')
plt.ylabel('Truth')

best_model.save('my_model.h5')





