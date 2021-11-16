import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import numpy as np

class PINN(tf.keras.models.Model):

    def __init__(self, func, input_shape: tuple, learning_rate: float, x_train: list):

        super(PINN, self).__init__()
        
        self.func = func

        self.x_train = x_train

        self.Model = tf.keras.Sequential()
        self.Model.add(tf.keras.layers.Dense(100, activation= tf.nn.relu, input_shape= input_shape))
        self.Model.add(tf.keras.layers.Dense(100, activation= tf.nn.relu))
        self.Model.add(tf.keras.layers.Dense(100, activation= tf.nn.relu))
        self.Model.add(tf.keras.layers.Dense(100, activation= tf.nn.relu))
        self.Model.add(tf.keras.layers.Dense(1))

        self.Model.compile(optimizer= tf.keras.optimizers.SGD(learning_rate= learning_rate, nesterov= True), loss= self.loss, loss_weights= [100, 1])

    def loss(self, y_true, y_pred):
        x = tf.reshape(self.x_train[:, 0], (len(self.x_train[:, 0]), 1))
        x_label = tf.reshape(self.x_train[:, 1], (len(self.x_train[:, 1]), 1))

        with tf.GradientTape(persistent= True) as g:
            g.watch(x)

            with tf.GradientTape(persistent= True) as gg:
                gg.watch(x)
                
                trueFunc = self.func(x)
                predFunc = self.Model(x, training= True)

        dTrueFunc = g.gradient(trueFunc, x)
        ddTrueFunc = gg.gradient(trueFunc, x)

        dPredFunc = g.gradient(predFunc, x)
        ddPredFunc = gg.gradient(predFunc, x)

        inside = tf.pow(tf.add(tf.cast(predFunc, tf.float64), ddPredFunc), 2)
        boundary = tf.pow(tf.subtract(tf.multiply(x, x_label), tf.multiply(trueFunc, x_label)), 2)

        return inside, boundary
