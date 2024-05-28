import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

dataset = pd.read_csv("Earthquake.csv")

dataset=dataset.drop(['time','dmin','net','place','type','horizontalError','depthError','magError','status','locationSource','id','updated','magSource','magNst'],axis=1)
print(dataset)

print(dataset.isnull().sum())

dataset['nst']=dataset['nst'].fillna(dataset['nst'].mean())
dataset['gap']=dataset['gap'].fillna(dataset['gap'].mean())
dataset['rms']=dataset['rms'].fillna(dataset['rms'].mean())

print(dataset.isnull().sum())

move=dataset.pop('mag')
dataset.insert(len(dataset.columns),'mag',move)
print(dataset)

X= dataset.iloc[:,:-1].values
y= dataset.iloc[:,-1].values

ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
X=np.array(ct.fit_transform(X))
print(X)

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=1)

sc=StandardScaler()
X_train[:,:-1]=sc.fit_transform(X_train[:,:-1])
X_test[:,:-1]=sc.fit_transform(X_test[:,:-1])

error=[]
r2=[]

lin_reg=  LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred1= lin_reg.predict(X_test)
error.append(round(mean_squared_error(y_test, y_pred1),2))
r2.append(round(r2_score(y_test, y_pred1),2))

svr_reg= SVR(kernel='linear')
svr_reg.fit(X_train ,y_train)
y_pred2= svr_reg.predict(X_test)
error.append(round(mean_squared_error(y_test, y_pred2),2))
r2.append(round(r2_score(y_test, y_pred2),2))

rf_reg= RandomForestRegressor(n_estimators=200, random_state=0)
rf_reg.fit(X_train, y_train)
y_pred3= rf_reg.predict(X_test)
error.append(round(mean_squared_error(y_test, y_pred3),2))
r2.append(round(r2_score(y_test, y_pred3),2))

print(error)
print(r2)


from sklearn.model_selection import cross_val_score
accuricies = cross_val_score(estimator= rf_reg, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuricies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuricies.std()*100))

plt.plot(pd.Series(y_test),color='r')
plt.plot(pd.Series(y_pred3),color='g')
plt.show()