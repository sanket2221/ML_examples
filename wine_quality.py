import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
from sklearn.linear_model import  LinearRegression , LogisticRegression
from sklearn.cluster import KMeans
import pandas as pd
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
#data pre-processing

dataset = pd.read_csv('C:/Users/Sanket Barhate/Downloads/red-wine-quality-cortez-et-al-2009/winequality_red.csv')
x = dataset.iloc[:,[0,1,3,4,5,6,7,8,9,10]].values
y = dataset.iloc[:,-1].values

#spliting into train and test
x_train , x_test, y_train, y_test = train_test_split(x, y, test_size=0.2 , random_state=0 )


#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

from sklearn.decomposition import PCA
pca = PCA(n_components= 3)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
covariance = pca.explained_variance_ratio_

#fitting multiple linear reg to traning set
regressor = LogisticRegression()
regressor.fit(x_train,y_train)
#print(regressor.coef_)
#predicting the test set result

y_pred = regressor.predict(x_test).astype(int)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

from sklearn.metrics import  confusion_matrix
cm = confusion_matrix(y_test, y_pred)


'''#optimal model using backward elimination
import statsmodels.api as sm
import statsmodels.regression.linear_model as smf
x = np.append(arr = np.ones((1599,1)).astype(int) ,values = x, axis= 1)
x_opt = x[:, [0,1,3,4,5,6,7,8,9,10]]
regressor_ols = sm.OLS(endog= y , exog= x_opt).fit()
print(regressor_ols.summary())

'''
