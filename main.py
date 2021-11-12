import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from PINNModel import PINN
from CustomLoss import customLoss



def main():

    x = tf.random.uniform((100,), minval= 0, maxval= 1)

    func = lambda y: tf.multiply(y, 2)

    target = func(tf.linspace(0, 1, 100))

    loss = customLoss(func)

    nn = PINN(lossFunc= loss.call)

    print(nn.Model.summary())

    res = nn.Model.fit(x, target, epochs= 100)

    plt.figure()
    plt.plot(nn.Model.predict(tf.linspace(0, 1, 100)))
    plt.plot(target)
    plt.legend(["Prediction", "Target"])
    plt.show()

if __name__ == "__main__":
    main()
    