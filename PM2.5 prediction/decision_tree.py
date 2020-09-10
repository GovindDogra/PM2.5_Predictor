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

from sklearn.tree import DecisionTreeRegressor
reg=DecisionTreeRegressor()
reg.fit(X_train,y_train)

print('the R^2 value of train set :',reg.score(X_train,y_train))
print('the R^2 value of test set :',reg.score(X_test,y_test))

from sklearn.model_selection import cross_val_score
c_v=cross_val_score(reg,X,y,cv=5)
print(c_v.mean())

#visualising decision tree
import pydotplus
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
features=list(df.columns[:-1])
import os
os.environ['PAth']=os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r'\Library\bin\graphviz'#craeting path where graphiviz liabrary is created
dot_data=StringIO()  #this var is created for telling where to get our output i.e O\P console
export_graphviz(reg,out_file=dot_data,feature_names=features,filled=True,rounded=True)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

#hyperparameter tuning
parameters={'splitter':['best','random'],
            'max_depth':[3,4,5,6,8,10,12,15,16],'min_samples_leaf':[1,2,3,4,5],
            'min_weight_fraction_leaf':[0.1,0.2,0.3,0.4],'max_features':['auto','sqrt','log2',None],
            'max_leaf_nodes':[None,10,20,40,50,70]}
from sklearn.model_selection import GridSearchCV
random_search=GridSearchCV(reg,param_grid=parameters,scoring='neg_mean_squared_error',n_jobs=1,cv=10,verbose=3)

random_search.fit(X,y)

print(random_search.best_params_)

#predicting test results 
Y_pred=random_search.predict(X_test)

sb.distplot(y_test-Y_pred)

#regression evaluation metrics
import sklearn.metrics as mt
print('MAE:',mt.mean_absolute_error(y_test,pred))
print('MSE:',mt.mean_squared_error(y_test,pred))
print('RMSE:',np.sqrt(mt.mean_absolute_error(y_test,pred)))

import pickle
file=open('decisiontree_model.pkl','wb')
pickle.dump(random_search,file)