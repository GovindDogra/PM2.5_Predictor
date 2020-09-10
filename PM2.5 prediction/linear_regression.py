import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

#importng dataset
df=pd.read_csv('Data/Real-Data/Real_Combine.csv')

#checking for null values 
sb.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')

df=df.dropna()

X=df.iloc[:,:-1] #independent var.
y=df.iloc[:,-1] #dependent var.

#feature selection(multivariate analysis)
sb.pairplot(df)

a=df.corr()#for checking corelations

#feature importance
from sklearn.ensemble import ExtraTreesRegressor
model= ExtraTreesRegressor()
model.fit(X,y)
print(model.feature_importances_)
#plotting the importance of  features
feat_imp=pd.Series(model.feature_importances_,index=X.columns)
feat_imp.nlargest(5).plot(kind='barh')#picking 5 columns that are in corelations with pm2.5
plt.show()

#splitting into test and train
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=0)

#model creation
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predicted vlaues
y_pred=regressor.predict(X_test)

print(regressor.coef_)#slope valuee of variables
coeff=pd.DataFrame(regressor.coef_,X.columns,columns=['coefficients'])#example,if coeff_ of T is -8.541.. ,that means
                                                                      #with increase in 1 unit of T ,-8.511.... unit of pm2.5 is decreasing
print(regressor.intercept_)#intercept,i.e when all our independent var. will be 0 we will get this value

#R-square vlaue for train and test set 
print('r^2 value for train set:',regressor.score(X_train,y_train))
print('r^2 value for test set:',regressor.score(X_test,y_test))

# Graphs
plt.scatter(y_test,y_pred)

sb.distplot(y_test-y_pred)

#regression evaluation metrics
import sklearn.metrics as mt
print('MAE:',mt.mean_absolute_error(y_test,y_pred))
print('MSE:',mt.mean_squared_error(y_test,y_pred))
print('RMSE:',np.sqrt(mt.mean_absolute_error(y_test,y_pred)))

import pickle
file=open('lin_regression_model.pkl','wb')
pickle.dump(regressor,file)












