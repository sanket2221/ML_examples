import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
from sklearn.linear_model import  LinearRegression
import pandas as pd

#data pre-processing

dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values
#encoding
labelencoder_x = LabelEncoder()
x[:,3] = labelencoder_x.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features= [3])
x = onehotencoder.fit_transform(x).toarray()
#avoiding dummy variable trap
x = x[:, 1:]

#spliting into train and test
x_train , x_test, y_train, y_test = train_test_split(x, y, test_size=0.2 , random_state=0 )

#fitting multiple linear reg to traning set
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#predicting the test set result

y_pred = regressor.predict(x_test)

#optimal model using backward elimination
import statsmodels.api as sm
import statsmodels.regression.linear_model as smf

x_opt = x[:, [0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog= y , exog= x_opt).fit()
print(regressor_ols.summary())
x_opt = x[:, [0,3]]
regressor_ols = sm.OLS(endog= y , exog= x_opt).fit()
print(regressor_ols.summary())