from numpy import gradient
import tensorflow as tf
import tensorflow.keras.backend as K

class customLoss(tf.keras.losses.Loss):
    def __init__(self, func) -> None:
        super().__init__()

        self.func = func

    def call(self, y_true, y_pred):

        with tf.GradientTape(persistent=True) as g:

            g.watch([y_pred, y_true])

            with tf.GradientTape(persistent=True) as gg:

                gg.watch([y_pred, y_true])
                
                trueFunc = self.func(y_true)
                predFunc = self.func(y_pred)

        dTrueFunc = g.gradient(trueFunc, y_true)
        ddTrueFunc = gg.gradient(trueFunc, y_true)

        dPredFunc = g.gradient(predFunc, y_pred)
        ddPredFunc = gg.gradient(predFunc, y_pred)

        loss = tf.reduce_mean(tf.pow(tf.subtract(trueFunc, predFunc), 2)) #+ tf.reduce_mean(tf.pow(tf.subtract(dTrueFunc, dPredFunc), 2))
            
        return loss
