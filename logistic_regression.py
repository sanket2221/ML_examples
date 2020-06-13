from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values
#print(pd.value_counts(y))

x_train ,x_test, y_train ,y_test = train_test_split(x,y, test_size= 0.25,random_state= 0)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

regressor = LogisticRegression(random_state= 0)
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

#making a confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#visualising the data
from matplotlib.colors import ListedColormap
x_set , y_set = x_train , y_train
x1, x2 = np.meshgrid(np.arange(start =x_set[:,0].min()-1 , stop = x_set[:,0].max()+1 ,step =0.01),
                     np.arange(start =x_set[:,1].min()-1 , stop = x_set[:,1].max()+1 ,step =0.01)
                    )
plt.contourf(x1 ,x2, regressor.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape)
            , alpha = 0.75 ,cmap =ListedColormap(('red', 'green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i ,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j,0],x_set[ y_set == j,1]
                ,c = ListedColormap(('red' ,'green'))(i),label =j)
plt.title('Logistic Regression')
plt.xlabel('Age')
plt.ylabel('Estimated salary')
plt.legend()
plt.show()