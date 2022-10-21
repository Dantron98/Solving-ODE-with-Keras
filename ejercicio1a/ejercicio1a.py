import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam
import matplotlib.pyplot as plt
import matplotlib
import math as m
import numpy as np
pi = tf.constant(m.pi)
matplotlib.use('TkAgg')



class ODEsolver(Sequential):
    loss_tracker = keras.metrics.Mean(name="loss")
    def train_step(self, data):
        x = tf.random.uniform((80, 1), minval=-5, maxval=5)

        with tf.GradientTape() as tape:
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                y_pred = self(x, training=True)
            x_o = tf.zeros((80, 1))
            y_o = self(x_o, training=True)
            y_1 = self(x_o, training=True)
            y_2 = self(x_o, training=True)
            ic = y_o - 3
            # ic1 = y_1
            # ic2 = y_2
            eq = 3 * keras.backend.sin(pi * x)
            loss = keras.losses.mean_squared_error(0., eq) + keras.losses.mean_squared_error(0., ic) #+ keras.losses.mean_squared_error(0., ic1) + keras.losses.mean_squared_error(0., ic2)

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
y = [(3*np.sin(np.pi*x)) for x in x_testv]

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
exit()
