import tensorflow as tf

from CustomLoss import customLoss

class PINN(tf.keras.models.Model):

    def __init__(self, lossFunc):

        super(PINN, self).__init__()

        self.lossFunc = lossFunc

        self.Model = tf.keras.Sequential()

        self.Model.add(tf.keras.layers.Dense(10, activation= tf.nn.relu, input_shape= (1, )))
        self.Model.add(tf.keras.layers.Dense(30, activation= tf.nn.relu))
        self.Model.add(tf.keras.layers.Dense(30, activation= tf.nn.tanh))
        self.Model.add(tf.keras.layers.Dense(1))

        self.Model.compile(optimizer= tf.keras.optimizers.SGD(learning_rate= 0.0001), loss= self.lossFunc)





