"""Utilities for Plotting."""

import matplotlib.pyplot as plt

def label_subplot(fig, ax, label, dx=0.005 * 4, dy=0.005):
    """
    Add a label with a fixed offset in figure coordinates to a subplot.

    Parameters
    ----------
    fig : plt.Figure
        Figure
    ax : plt.Axes()
        Axis corresponding to individual subplot.
    text : str
        Label
    dx, dy : float
        Amount to translate the letter relative to the top right corner of the subplot.
    """
    # Compose transformer:
    tf = ax.transAxes.inverted() + fig.transFigure
    # transFigure takes you from [0,1] figure coordinates to pixels
    # transAxes.inverted takes you from pixels to coordinates of axes
    # ax.text uses these coordinates

    # Note by default, text uses subplot coordinates, which gives undesirable
    # behavior of the subplots have different sizes.

    # Transform delta
    u, v = tf.transform((dx, dy)) - tf.transform((0, 0))
    ax.text(-u, 1 + v, label, size=12,
            #  weight='bold',
            transform=ax.transAxes)


def equalize_y_axes(ax0, ax1):
    u0, v0 = ax0.get_ylim()
    u1, v1 = ax1.get_ylim()

    u, v = min(u0, u1), max(v0, v1)
    for ax in [ax0, ax1]:
        ax.set_ylim(u, v)


def expand_legend_linewidths(ax, lw=3, **kwargs):
    # obtain the handles and labels from the figure
    handles, labels = ax.get_legend_handles_labels()
    # copy the handles
    import copy
    handles = [copy.copy(ha) for ha in handles]
    # set the linewidths to the copies
    [ha.set_linewidth(lw) for ha in handles]
    # put the copies into the legend
    leg = ax.legend(
        handles=handles, labels=labels,
        **kwargs)
    return leg
