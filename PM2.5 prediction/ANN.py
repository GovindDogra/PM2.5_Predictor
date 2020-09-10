import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

df=pd.read_csv('Data/Real-Data/Real_combine.csv')

df=df.dropna()

sb.pairplot(df)
df.corr()

X=df.iloc[:,:-1]
y=df.iloc[:,-1]

from sklearn.tree import ExtraTreeRegressor
et=ExtraTreeRegressor()
et.fit(X,y)
print(et.feature_importances_)
feat_imp=pd.Series(et.feature_importances_,index=X.columns)
feat_imp.nlargest(5).plot(kind='barh')
plt.show()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30)

from keras.models import Sequential
from keras.layers import Dense

regressor=Sequential()
#adding input layer and first hidden layer
regressor.add(Dense(units=128,kernel_initializer='normal',input_dim=X_train.shape[1],activation='relu'))

#adding 2nd hidden layer
regressor.add(Dense(units=256,kernel_initializer='normal',activation='relu'))

#adding 3rd hidden layer
regressor.add(Dense(units=256,kernel_initializer='normal',activation='relu'))

#adding 4th hidden layer
regressor.add(Dense(units=256,kernel_initializer='normal',activation='relu'))

#adding output layer
regressor.add(Dense(1,kernel_initializer='normal',activation='linear'))#as it is a linear reg. problem so taking activation = linear

#compiling the ANN
regressor.compile(optimizer='adam',loss='mean_absolute_error',metrics=['mean_absolute_error'])
regressor.summary()
regressor.fit(X_train,y_train,batch_size=10,epochs=100,validation_split=0.33)

pred=regressor.predict(X_test)

#regression evaluation metrics
import sklearn.metrics as mt
print('MAE:',mt.mean_absolute_error(y_test,pred))
print('MSE:',mt.mean_squared_error(y_test,pred))
print('RMSE:',np.sqrt(mt.mean_absolute_error(y_test,pred)))

