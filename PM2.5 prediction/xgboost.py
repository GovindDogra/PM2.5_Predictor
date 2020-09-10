import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

df=pd.read_csv('Data/Real-Data/Real_combine.csv')

sb.heatmap(df.isnull(),yticklabels=False,cmap='viridis')
#dropping null values
df=df.dropna()

sb.pairplot(df)
#checking correlation
a=df.corr()

#splitting datasets
X=df.iloc[:,:-1]
y=df.iloc[:,-1]

#feature selection
from sklearn.tree import ExtraTreeRegressor
model=ExtraTreeRegressor()
model.fit(X,y)
print(model.feature_importances_)
feat_imp=pd.Series(model.feature_importances_,index=X.columns)
feat_imp.nlargest(5).plot(kind='barh')#picking 5 columns that are in corelations with pm2.5
plt.show()

#splitting datasets into train and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

#applying xgboost regression
import xgboost as xgb
reg=xgb.XGBRegressor()
reg.fit(X_train,y_train)
y_pred=reg.predict(X_test)
print('R^2 value(train set):',reg.score(X_train,y_train))
print('R^2 value(test set):',reg.score(X_test,y_test))

#cross_validation
from sklearn.model_selection import cross_val_score
c_v=cross_val_score(reg,X,y,scoring='neg_mean_squared_error',cv=5,n_jobs=-1)
print(c_v)
c_v.mean()

#hyperparameter tuning (using randomizedsearchCV)
n_estimators=[int(x) for x in np.linspace(start=100,stop=1200,num=12)]
learning_rate=[0.1,0.3,0.4,0.5,0.7]
max_depth=[int(x) for x in np.linspace(start=1,stop=20,num=5)]
subsample=[0.5,0.7,0.8,0.9]
min_child_weight=[3,4,5,6,7]

parameters={'n_estimators':n_estimators,'learning_rate':learning_rate,'max_depth':max_depth,
            'subsample':subsample,'min_child_weight':min_child_weight}
from sklearn.model_selection import RandomizedSearchCV
rand_cv=RandomizedSearchCV(reg,param_distributions=parameters,n_iter=100,n_jobs=-1,scoring='neg_mean_squared_error',cv=5)
rand_cv.fit(X_train,y_train)

print(rand_cv.best_params_)
print(rand_cv.best_score_)
pred=rand_cv.predict(X_test)

#plotting
sb.distplot(y_test-pred)

#regression evaluation metrics
import sklearn.metrics as mt
print('MAE:',mt.mean_absolute_error(y_test,pred))
print('MSE:',mt.mean_squared_error(y_test,pred))
print('RMSE:',np.sqrt(mt.mean_absolute_error(y_test,pred)))

