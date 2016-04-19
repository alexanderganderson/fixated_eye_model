"""Code to generate a block prior."""

import numpy as np


def block_prior(m):
    """
    Create a dictionary implementing a block prior.

    Parameters
    ----------
    m : int
        Linear dimension of block prior image.
    Returns
    -------
    d : array, shape (n_l = m ** 2, n_pix = 4 * m ** 2)
        Dictionary that upsamples the image by a factor of 2.
    """
    n = 2 * m
    d = np.zeros((m, m, n, n))
    for i in range(m):
        for j in range(m):
            ii = 2 * i
            jj = 2 * j
            d[i, j, ii:ii + 2, jj:jj + 2] = 1
    return d.reshape(m * m, n * n)

if __name__ == '__main__':
    from utils.rf_plot import show_fields
    import matplotlib.pyplot as plt
    d = block_prior(7)
    show_fields(d, pos_only=True)
    plt.show()
