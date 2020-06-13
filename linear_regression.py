import numpy as np
import tensorflow as tf
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#spliting into train and test
from sklearn.model_selection import train_test_split

x_train , x_test, y_train, y_test = train_test_split(x, y, test_size=1/3 , random_state=0 )

#fitting simple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train , y_train)

#predicting the test set result
y_pred = regressor.predict(x_test)

import pickle
pickle.dump(regressor, open('webpage/regressor.pkl', 'wb'))
#visualising data
plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_train,regressor.predict(x_train),color = 'black')
plt.title('Salary vs Experience(training set )')
plt.xlabel('Years of experince')
plt.ylabel('Salary')
plt.show()
