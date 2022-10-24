import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam
import matplotlib.pyplot as plt
import matplotlib
import math as m
matplotlib.use('TkAgg')
import numpy as np
#

pi = tf.constant(m.pi)
class ODEsolver(Sequential):
    loss_tracker = keras.metrics.Mean(name="loss")
    def train_step(self, data):
        batch_size = tf.shape(data)[0]
        x = tf.random.uniform((batch_size, 1), minval=-1, maxval=1)

        with tf.GradientTape() as tape:
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                y_pred = self(x, training=True)
            dy = tape2.gradient(y_pred, x)
            x_o = tf.zeros((batch_size, 1))
            x_1 = x_o + 1
            x_2 = -x_1
            y_o = self(x_o, training=True)
            y_1 = self(x_2, training=True)
            y_2 = self(x_1, training=True)
            eq = dy - 2 - 12*x**2
            ic = y_o - 1
            ic2 = y_1 + 5
            ic3 = y_2 - 7

            loss = 2*keras.losses.mean_squared_error(0., eq) + keras.losses.mean_squared_error(0., ic) + keras.losses.mean_squared_error(0., ic2) + keras.losses.mean_squared_error(0., ic3)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [keras.metrics.Mean(name='loss')]

model = ODEsolver()

model.add(Dense(10, activation='tanh', input_shape=(1,)))
model.add(Dense(1, activation='tanh'))
model.add(Dense(1, activation='linear'))
model.summary()

model.compile(optimizer=RMSprop(), metrics=['loss'])
tf.keras.layers.Dropout(.25, input_shape=(2,))
x = tf.linspace(-1, 1, 1000)
history = model.fit(x, epochs=500, verbose=1)

x_testv = tf.linspace(-1, 1, 1000)
y = [(1 + 2*x + 4*x**3) for x in x_testv]

a = model.predict(x_testv)
plt.grid()
plt.title('Solución encontrada por la red vs solución analitica')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_testv, a, '-', label='Solución de la red')
plt.plot(x_testv, y, label='Solución análitica')
plt.legend()
plt.savefig('commit1.png')
plt.show()
model.save('red2.1.h5')
