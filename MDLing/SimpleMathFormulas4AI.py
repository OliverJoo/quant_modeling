import numpy as np
import numpy_financial as npf
import pandas as pd
import tensorflow as tf


# ddof = 0 : sample variance
def beta_linear_regression(x, y):
    """
        slopoe of linear regression
    :param x: feature value
    :param y: label value
    :return: beta
    """
    return np.cov(x, y, ddof=0)[0, 1] / np.var(x)


def alpha_linear_regression(x, y):
    """
        intercept of linear regression
    :param x:
    :param y:
    :return:
    """
    return y.mean() - (beta_linear_regression(x, y) * x.mean())


def MSE(y, pred_y):
    """
        mean squared error
    :param y: real y
    :param pred_y: estimated y
    :return: MSE
    """
    return ((y - pred_y) ** 2).mean()


def relu(x):
    return np.maximum(0, x)


def swish(x):
    return x * tf.nn.sigmoid(x)


def mish(x):
    return x * tf.nn.tanh(tf.nn.softplus(x))
