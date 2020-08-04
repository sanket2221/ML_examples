import tensorflow as tf
from tensorflow.python.keras.losses import mean_squared_error

tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np

c = np.array([-40, -10, 0,10,15], dtype=float)
f = np.array([-40, 14, 32,50,59], dtype=float)

 
#10 = tf.keras.layers.Dense(units=1, input_shape=[1])

model = tf.keras.Sequential([
tf.keras.layers.Dense(units=5, input_shape=[1]),
tf.keras.layers.Dense(units=5, input_shape=[1]),
tf.keras.layers.Dense(units=1, input_shape=[1])
])

model.compile(loss='mean_squared_error' , optimizer=tf.keras.optimizers.Adam(0.1))
history =  model.fit(c,f,epochs=3600,verbose=False)
print("Finished Training Model ")
model.save('celcius.model')
print(model.predict([100.0]))