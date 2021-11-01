"Function to plot the receptive fields of a matrix."""

import numpy as np
import matplotlib.pyplot as plt


def show_fields(
    d,
    cmap=plt.cm.gray,
    m=None,
    pos_only=False,
    colorbar=True,
    fig=None, ax=None,
    alpha=1.,
    normed=False
):
    """
    Plot a collection of images.

    Parameters
    ----------
    d : array, shape (n, n_pix)
        A collection of n images unrolled into n_pix length vectors
    cmap : plt.cm
        Color map for plot
    m : int
        Plot a m by m grid of receptive fields
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1)
    n, n_pix = d.shape
    if m is None:
        m = int(np.sqrt(n - 0.01)) + 1

    if normed:
        d = d / abs(d).max(axis=1, keepdims=True)

    l = int(np.sqrt(n_pix))  # Linear dimension of the image

    mm = np.max(np.abs(d))

    out = np.zeros(((l + 1) * m - 1, (l + 1) * m - 1)) + mm

    for u in range(n):
        i = u / m
        j = u % m
        out[(i * (l + 1)):(i * (l + 1) + l),
            (j * (l + 1)):(j * (l + 1) + l)] = np.reshape(d[u], (l, l))

    if pos_only:
        m0 = 0
    else:
        m0 = -mm
    m1 = mm
    cax = ax.imshow(out, cmap=cmap, interpolation='nearest', vmin=m0, vmax=m1,
                    alpha=alpha)
    if colorbar:
        fig.colorbar(cax, ax=ax)
    ax.set_axis_off()
