# Util functions (mostly gd) for the problem notebooks (mostly 1 and 2)
import numpy as np
import matplotlib.pyplot as plt 

# creates feature map for quadratic regression
def feature_map_quadratic(x):
    return np.vstack([x**2, x, np.ones(len(x))]).T

# creates feature map for quadratic regression
def feature_map_linear(x):
    return np.vstack([x, np.ones(len(x))]).T

# function for gradient descent
def gradient_descent(X, y, learning_rate=0.1, iters=1000):
    m, n = X.shape
    w = np.zeros(n)

    for _ in range(iters):
        y_pred = X.dot(w)
        gradient = X.T.dot(y_pred - y) / m
        w -= learning_rate * gradient
    return w

# scatter plots the points x_scatter and y_scatter, and plots the function x_pred and y_pred
def plot(x_scatter, y_scatter, x_pred, y_pred, pred_label, scatter_label, plot_label):
    plt.plot(x_pred, y_pred, label=pred_label)
    plt.scatter(x_scatter, y_scatter, color="red", label=scatter_label)
    plt.title(plot_label)
    plt.legend()
    plt.show()
