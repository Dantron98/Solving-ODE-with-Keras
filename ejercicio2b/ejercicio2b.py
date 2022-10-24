import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
#Hoola mundo 3
class ODEsolver(Sequential):
    loss_tracker = keras.metrics.Mean(name="loss")

    def train_step(self, data):
        batch_size = tf.shape(data)[0]
        x = tf.random.uniform((batch_size, 1), minval=-5, maxval=5)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                y_pred = self(x, training=True)
            dy = tape2.gradient(y_pred, x)
            dy2 = tape.gradient(dy, x)
            x_0 = tf.zeros((batch_size, 1))
            y_0 = self(x_0, training=True)
            y_1 = self(x_0, training=True)
            eq = dy2 + y_pred
            ic = y_0 - 1
            ic2 = dy + 0.5
            loss = keras.losses.mean_squared_error(0., eq) + (1/3)*keras.losses.mean_squared_error(0., ic) + (.25/900)*keras.losses.mean_squared_error(0., ic2)

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
x = tf.linspace(-5, 5, 1000)
history = model.fit(x, epochs=1000, verbose=1)

x_testv = tf.linspace(-5, 5, 1000)
y = [(np.cos(x)-0.5*np.sin(x)) for x in x_testv]

a = model.predict(x_testv)
plt.grid()
plt.title('Solución encontrada por la red vs solución analitica')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_testv, a, '-', label='Solución de la red')
plt.plot(x_testv, y, label='Solución análitica')
plt.legend()
plt.savefig('commit2.png')
plt.show()

model.save('red2.1.h5')
exit()

model.save('red2.h5')
modelo_cargado = tf.keras.models.load_model('red5.h5')