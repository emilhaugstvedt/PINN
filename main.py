from numpy.core.fromnumeric import shape, size
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from PINNModel import PINN
from plot import plot, plotHistory

def main():

    # Defining the sampling space:
    numSamples = 100
    minSample = 0.0
    maxSample = np.pi/2

    # Function to estimate:
    func = lambda x: tf.sin(x)

    # Sample random for trainig:
    x_train = np.random.uniform(low= 0, high= 3, size= (100 - 2, 2))
    x_train[:, 1] = np.zeros(98)
    x_train = np.append(x_train, np.array([[0, 0], [np.pi/2, 2]]), axis= 0)

    x_train = tf.convert_to_tensor(x_train)

    # Sample for predicting:
    x_pred = tf.linspace(minSample, maxSample, numSamples)

    # Target value for training:
    target = func(x_pred)

    # Create neural network:
    nn = PINN(func= func, input_shape= (1,), learning_rate= 0.0001, x_train= x_train)

    # Fit model:
    res = nn.Model.fit(x_train[:, 0],
                       target,
                       epochs= 200)

    # Make prediction:
    prediction = nn.Model.predict(x_pred)
    true = func(x_pred)

    print(prediction.shape)

    # Plot prediction vs true and training history:
    plot(prediction= 100*prediction, true= true)
    plotHistory(res)

if __name__ == "__main__":
    main()
    