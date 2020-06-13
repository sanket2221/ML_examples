import pandas as pd
import tensorflow as tf

#preprocessing

dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values


from sklearn.preprocessing import OneHotEncoder , LabelEncoder
lencoder_x1 = LabelEncoder()
x[:,1] = lencoder_x1.fit_transform(x[:,1])
lencoder_x2 = LabelEncoder()
x[:,2] = lencoder_x2.fit_transform(x[:,2])
hot_encoder = OneHotEncoder(categorical_features= [1])
x = hot_encoder.fit_transform(x).toarray()
x= x[:,1:]

from sklearn.model_selection import train_test_split
x_train , x_test, y_train, y_test = train_test_split(x, y, test_size=0.2 , random_state=0)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

hidden_layer_size = 6
input_layer_size = 11

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(output_dim = 10 , activation='relu',init = 'uniform', input_dim= 11))

classifier.add(Dense(output_dim = 10 , activation='relu',init = 'uniform'))

classifier.add(Dense(output_dim = 1 , activation='sigmoid',init = 'uniform'))

classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'] )

classifier.fit(x_train , y_train , batch_size= 10 ,epochs=100,validation_data=(x_test, y_test))

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
