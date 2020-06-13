from sklearn.datasets import make_blobs
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

data = make_blobs(n_samples=2000 , n_features=20 , centers=5)

df = pd.DataFrame(data[0])
print(data[1])


scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
'''
input_size = 5
output_size = input_size


l1 = tf.keras.layers.Dense(input_size, activation='relu')
l2 = tf.keras.layers.Dense(5, activation='relu')
ae = tf.keras.layers.Dense(2, activation='relu',name='ae')
l3 = tf.keras.layers.Dense(5,activation='relu')
l4 = tf.keras.layers.Dense(output_size,activation='softmax')

model = tf.keras.Sequential([l1, l2, ae, l3, l4] )

model.compile(optimizer = 'adam' , loss='mean_squared_error' , metrics=['accuracy'])
model.fit(x= scaled_data,y = scaled_data , epochs=50 )

encoder = Model(model.input , model.get_layer('ae').output)

data_reduced = encoder.predict(scaled_data) '''

m = Sequential()
m.add(Dense(20,  activation='elu', input_shape=(20,)))
m.add(Dense(10,  activation='elu'))
m.add(Dense(2,    activation='linear', name="bottleneck"))
m.add(Dense(10,  activation='elu'))
m.add(Dense(20,  activation='sigmoid'))
m.compile(loss='mean_squared_error', optimizer = Adam())
history = m.fit(scaled_data, scaled_data, batch_size=128, epochs=20, verbose=1)

encoder = Model(m.input, m.get_layer('bottleneck').output)
data_enc = encoder.predict(scaled_data)  # bottleneck representation
data_dec = m.predict(scaled_data)        # reconstruction

plt.scatter(data_enc[:,0], data_enc[:,1], c=data[1][:], s=8, cmap='tab10')
plt.show()
plt.tight_layout()

