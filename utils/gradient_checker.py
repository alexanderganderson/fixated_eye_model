# gradient_checker.py

# Functions to check the derivative and hessian of a function numerically

import numpy as np


def gradient_check(f, fp, input_shape, x0=None, y0=None):
    """
    Function to check if fp is the gradient of f stochastically.
    Returns the derivative of f calculated two ways

    Parameters
    ----------
    f : function, input_shape -> float
        A function taking in a number and outputting a number
    fp : function, input_shape -> input_shape
        Derivative of f
    input_shape : tuple
        Shape of the input to f
    x0 : numpy array, shape = input_shape
        Point to evaluate gradient, if None, chosen randomly
    y0 : numpy array, shape = input_shape
        Direction to evaluate gradient, if None, chosen small relative to x0

    Returns
    -------
    u : array, shape input_shape
        Derivative calculated by taking differences of f
    v : array, shape input_shape
        Derivative calculated using fp
    """
    x0, y0, y0_norm = prep_point_and_direction(x0, y0, input_shape)
    u = (f(x0 + y0 * 0.5) - f(x0 - y0 * 0.5)) / y0_norm
    v = (y0 * fp(x0)).sum() / y0_norm
    return u, v


def hessian_check(f, fpp, input_shape, x0=None, y0=None):
    """
    Function to check if fpp is the hessian of f

    f : function

    fpp : hessian of f

    input_shape : tuple
        Shape of the input to f and fpp, must be in form (N,)
    """
    x0, y0, y0_norm = prep_point_and_direction(x0, y0, input_shape)
    u = ((f(x0 + y0) - f(x0)) - (f(x0) - f(x0 - y0)))/y0_norm ** 2
    v = np.einsum('i, ij, j', y0, fpp(x0), y0) / y0_norm ** 2
    return u, v


def prep_point_and_direction(x0, y0, input_shape):
    if x0 is None:
        x0 = np.random.randn(*input_shape)
    if y0 is None:
        m = np.max(np.abs(x0))
        y0 = m * 0.01 * np.random.randn(*input_shape)
    y0_norm = np.sqrt((y0 ** 2).sum())
    return x0, y0, y0_norm

if __name__ == '__main__':
    d = 10
    input_shape = (d,)

    def f(x):
        return 0.5 * (x ** 2).sum()

    def fp(x):
        return x

    def fpp(x):
        return np.eye(x.size)

    u, v = gradient_check(f, fp, input_shape)
    print 'The gradient check gave {}, {}'.format(u, v)

    u, v = hessian_check(f, fpp, input_shape)
    print 'The hessian check gave {}, {}'.format(u, v)
