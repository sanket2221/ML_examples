import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
from sklearn.linear_model import  LinearRegression
import pandas as pd

#data pre-processing

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values


#linear_regresion
lin_reg = LinearRegression()
lin_reg.fit(x,y)

#polynomial_regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

#visualising data linear reg
plt.scatter(x,y)
plt.plot(x, lin_reg.predict(x),color = 'blue')
plt.title('linear regression')
plt.xlabel('position')
plt.ylabel('salaries')

#visualising data poly reg
plt.scatter(x,y, color = 'red')
plt.plot(x, lin_reg2.predict(x_poly),color = 'green')
plt.title('polynomial regression')
plt.xlabel('position')
plt.ylabel('salaries')

