import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

#importng dataset
df=pd.read_csv('Data/Real-Data/Real_Combine.csv')

df.head()

df=df.dropna()

X=df.iloc[:,:-1] #independent var.
y=df.iloc[:,-1] #dependent var.

#splitting into test and train
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=0)


#linear regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
lin_regg=LinearRegression()
mse=cross_val_score(lin_regg,X,y,scoring='neg_mean_squared_error',cv=5)
mean_mse=np.mean(mse)
print(mean_mse)

#ridge regression 
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
regg=Ridge()
parameters={'alpha':[1e-20,1e-15,1e-10,1e-5,1e-3,1e-2,1,5,10,20,30,50,100]}
rid_regg=GridSearchCV(regg,parameters,scoring='neg_mean_squared_error',cv=5)
rid_regg.fit(X,y)
print(rid_regg.best_score_)
print(rid_regg.best_params_)


#lasso reg
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
regressor=Lasso()
parameters={'alpha':[1e-20,1e-15,1e-10,1e-5,1e-3,1e-2,1,5,10,20,30,50,100]}
lasso_regg=GridSearchCV(regressor,parameters,scoring='neg_mean_squared_error',cv=5)
lasso_regg.fit(X,y)
print(lasso_regg.best_params_)
print(lasso_regg.best_score_)

ridge_pred=rid_regg.predict(X_test)
lasso_pred=lasso_regg.predict(X_test)

#ridge graphs
sb.distplot(y_test-ridge_pred)
plt.scatter(y_test,ridge_pred)


#lasso graphs
sb.distplot(y_test-lasso_pred)
plt.scatter(y_test,lasso_pred)


#regression evaluation metrics
import sklearn.metrics as mt
print('MAE:',mt.mean_absolute_error(y_test,lasso_pred))
print('MSE:',mt.mean_squared_error(y_test,lasso_pred))
print('RMSE:',np.sqrt(mt.mean_absolute_error(y_test,lasso_pred)))


import pickle
file=open('ridge_lasso_model.pkl','wb')
pickle.dump(lasso_regg,file)






