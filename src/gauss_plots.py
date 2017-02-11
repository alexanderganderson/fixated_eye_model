import numpy as np
import matplotlib.pyplot as plt

from src.analyzer import snr

from scipy.signal import gaussian, convolve2d

def smooth_image(s):
    """
    Smooth the image using the 0.5 ds gauss blur.

    Parameters
    ----------
    s : array, shape (l_i, l_i)
        Unraveled image.

    Returns
    -------
    sp : array, shape (n_pix,)
        Image after blurring.
    """
    f = gaussian(9, 0.5)
    f = np.einsum('i,j->ij', f, f)
    return convolve2d(s, f, mode='same')

def plot_image(s, ds, colorbar=False, alpha=1.,
                    cmap=plt.cm.gray, title='', vmin=None, vmax=None):
    """
    Plot the image.

    Parameters
    ----------
    s : array, shape (l_i, l_i)
        Unraveled image to be plotted.
    ds : float
        Spacing between pixels.
    colorbar : bool
        If true, show a colorbar.
    alpha : float
        Alpha for showing image.
    cmap : plt.cm
        Colormap.
    title : str
        String for the title.
    vmin, vmax : float
        Min and max for imshow.
    """
    s1 = smooth_image(s)
    l_i = s.shape[0]
    if vmin is None:
        vmin = s.min()
    if vmax is None:
        vmax = s.max()
    a = ds * l_i / 2
    plt.imshow(s1, cmap=cmap, interpolation='gaussian',
               extent=[-a, a, -a, a], alpha=alpha,
               vmin=vmin, vmax=vmax)
    plt.title(title)
    if colorbar:
        plt.colorbar()


def plot_rfs(ax, xe, ye, de, rf_ratio, legend=False, alpha=0.1):
    """
    Plot the image and the receptive fields.

    Parameters
    ----------
    ax : plt.axes
        Axes for plotting.
    xe, ye: array, shape (n_n,)
        Neuron RF centers
    xr, yr: array, shape (1, n_t)
        Location of eye at time t
    de: float
        Neuron spacing
    """
    #  ax = plt.axes()
    ax.set_aspect('equal')
    # FIXME: HARD CODED 2x
    r = rf_ratio * de
    for i, (x, y) in enumerate(zip(xe, ye)):
        if i == 0:
            label = 'One SDev of Neuron RF'
        else:
            label = None
        ax.add_patch(plt.Circle((x, y), r, color='green', fill=True,
                                alpha=alpha,
                                label=label))
    if legend:
        plt.legend()
        plt.xlabel('x (arcmin)')
        plt.ylabel('y (arcmin)')


def plot_fourier_spectrum(ax, s, ds, de):
    """
    Plot the fourier spectrum.

    Parameters
    ----------
    ax : plt.axes
        Plotting axes.
    s : array, shape (l_i, l_i)
        Image to get the fourier transform of.
    ds : float
        Spacing of the pixels.
    de : float
        Spacing of Receptive fields.
    """
    s1 = smooth_image(s)
    l_i = s.shape[0]
    plt.title('Fourier Transform')
    freqs = np.fft.fftshift(np.fft.fftfreq(l_i)) / ds
    df = (freqs[1] - freqs[0])
    fmin = freqs[0] - df / 2
    fmax = freqs[-1] + df / 2
    plt.imshow(np.abs(np.fft.fftshift(np.fft.fft2(s1))),
               extent=[fmin, fmax, fmin, fmax],
               cmap=plt.get_cmap('afmhot'))
    fn = 0.5 * de
    ax.add_patch(plt.Circle((0, 0), fn, color='gray', alpha=0.5))
    ax.add_patch(plt.Circle((0, 0), fn / (np.sqrt(3.) / 2.), color='gray', alpha=0.25))
    plt.colorbar()


def compare_fourier(s_gen, s_inf, l_i, ds, de,
                    xe, ye, rf_ratio):
    """
    Plot the two fourier spectrums.

    Parameters
    ----------
    s_gen : array, shape (l_i, l_i)
        Image that generated the observations.
    s_inf : array, shape (l_i, l_i)
        Inferred image.
    ds : float
        Spacing between pixels.
    de : float
        Spacing between receptors.
    xe, ye : array, shape (n_sensors,)
        X, Y positions of sensors.
    """
    n_row, n_col = 2, 3
    fig, axes = plt.subplots(nrows=n_row, ncols=n_col,
                           figsize=(n_col * 6, n_row * 4.5))
    vmin, vmax = s_gen.min(), s_gen.max()

    for i, (s, name) in enumerate(zip([s_gen, s_inf], ['Original', 'Estimate'])):
        ax = plt.subplot(n_row, n_col, i * n_col + 1)
        plot_image(s, ds=ds, title='Image: {}'.format(name),
                   vmin=vmin, vmax=vmax)

        ax = plt.subplot(n_row, n_col, i * n_col + 2)
        plot_fourier_spectrum(ax, s, ds, de)

        ax = plt.subplot(n_row, n_col, i * n_col + 3)
        if i == 0:
            plot_rfs(ax, xe, ye, de, rf_ratio, alpha=0.5)
            plot_image(
                s_gen, ds=ds,
                title='Image: {}'.format('Image with RFs')
            )

def plot_path_estimate(est_mean, est_sdev, xyr, d, q, dt=0.001):
    """
    Plot the actual and estimated path generated.

    Parameters
    ----------
    q : int
        EM iteration number
    d : int
        Dimension to plot (either 0 or 1)
    """
    n_t = est_mean.shape[0]
    path = xyr[d]
    if (d == 0):
        label = 'Hor.'
    elif (d == 1):
        label = 'Ver.'
    else:
        raise ValueError('d must be either 0 or 1')

    tt = dt * np.arange(n_t)
    plt.fill_between(tt,
                     est_mean[:, d] - est_sdev[:, d],
                     est_mean[:, d] + est_sdev[:, d],
                     alpha=0.5, linewidth=1.)
    plt.plot(tt,
             est_mean[:, d], label='estimate')
    plt.plot(tt,
             path, label='actual')
    plt.xlabel('Time (s)')
    plt.ylabel('Relative position (arcmin)')
    plt.title('{} Position'.format(label))



def snr_one_iteration(s_gen, s_est, xyr, xyr_est, xs, ys, t, var):
    """
    Get the SNR between the two images, correcting for a shift.

    Parameters
    ----------
    s_gen : array, shape (l_i, l_i)
        True image.
    s_est : array, shape (l_i, l_i)
        Estimated image.
    xyr :
    xyr_est :
    xs :
    ys :
    t :
    var :

    Returns
    -------

    """
    xr, yr = xyr
    xr_est = xyr_est[:, 0]
    yr_est = xyr_est[:, 1]
    t += 1
    dx = np.mean(xr[0:t] - xr_est[0:t])
    dy = np.mean(yr[0:t] - yr_est[0:t])
    i1 = s_gen.ravel()
    i2 = s_est.ravel()
    return snr(i1, xs, ys,
               i2, xs + dx, ys + dy,
               var)

def get_snr_list(s_gen, estimates, xyr, xyr_est, xs, ys, ds):
    var = (0.5 * ds) ** 2
    return [snr_one_iteration(s_gen, s_est, xyr, xyr_est, xs, ys, q, var=var)
            for q, s_est in enumerate(estimates)]







