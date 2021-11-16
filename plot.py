import matplotlib.pyplot as plt
import tensorflow as tf

def plot(prediction: list, true: list):
    plt.figure()
    plt.plot(true)
    plt.plot(prediction)
    plt.legend(["Target", "Prediction"])
    plt.show()

def plotHistory(result):
    plt.figure()
    plt.plot(result.history["loss"])
    plt.show()