import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def logistic_func(x, x0, L, k):
    return L / (1 + np.exp(k * (x - x0)))


def morse_func(x, x0, D, A):
    return D * (np.exp(-A * (x - x0)) - 1) ** 2


def ln_morse_func(x, x0, D, A):
    return np.log(D) + 2 * (np.exp(-A * (x - x0)) - 1)


