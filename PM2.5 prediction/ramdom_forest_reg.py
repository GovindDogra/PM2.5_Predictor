#Importing Libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

#Importing dataset
df=pd.read_csv('Data/Real-Data/Real_combine.csv')

#cgecking for null values
sb.heatmap(df.isnull(),yticklabels=False,cmap='viridis')

#droping null values
df=df.dropna()

#plotting pairwise relationship between dataset
sb.pairplot(df)

#checking correlation
a=df.corr()

#splitting datasets
X=df.iloc[:,:-1]
y=df.iloc[:,-1].values
y=y.reshape(-1,1)

#handling missing vlaues(0 in pm2.5)
from sklearn.impute import SimpleImputer
im=SimpleImputer(missing_values=0,strategy='mean')
im=im.fit(y)
y=im.transform(y)

#feature selection
from sklearn.tree import ExtraTreeRegressor
model=ExtraTreeRegressor()
model.fit(X,y)
print(model.feature_importances_)
feat_imp=pd.Series(model.feature_importances_,index=X.columns)
feat_imp.nlargest(5).plot(kind='barh')#picking 5 columns that are in corelations with pm2.5
plt.show()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

from sklearn.ensemble import RandomForestRegressor
reg=RandomForestRegressor()
reg.fit(X_train,y_train)
y_pred=reg.predict(X_test)

print('r^2 value of train set:',reg.score(X_train,y_train))
print('r^2 value of test set:',reg.score(X_test,y_test))
#plotting
sb.distplot(y_test-y_pred)

from sklearn.model_selection import cross_val_score
cv=cross_val_score(reg,X,y,cv=5,n_jobs=-1)
print(cv)
cv_mean=cv.mean()

#hyperparameter tuning
n_estimators=[int(x) for x in np.linspace(start=100,stop=1200,num=12)]
max_depth=[int(x) for x in np.linspace(start=5,stop=30,num=6)]
parameters={'n_estimators':n_estimators,'max_features':['auto','sqrt'],
            'max_depth':max_depth,'min_samples_split':[2,5,10,15,20],'min_samples_leaf':[1,2,5,10]}
from sklearn.model_selection import RandomizedSearchCV
random_search=RandomizedSearchCV(reg,param_distributions=parameters,n_iter=100,scoring='neg_mean_squared_error',cv=5,verbose=2)
random_search.fit(X_train,y_train)

random_search.best_params_
random_search.best_score_

pred=random_search.predict(X_test)
#plotting
sb.distplot(y_test-pred)

#regression evaluation metrics
import sklearn.metrics as mt
print('RMSE:',np.sqrt(mt.mean_absolute_error(y_test,pred)))

import pickle
file=open('randomforest_model.pkl','wb')
pickle.dump(random_search,file)

#for sinagle day
random_search.predict([[24.7 , 29.9 , 20.5 , 1018.5 , 65.0 , 6.9 , 8.0 , 14.8]])
