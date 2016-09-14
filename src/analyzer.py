"""Code to analyze the output of the simulations."""

import numpy as np
import cPickle as pkl
import sys
import os
# from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt


def inner_product(p1, l1x, l1y,
                  p2, l2x, l2y, var):
    """
    Calculate the inner product between two images.

    Image representation:
    p1 -> array of pixels
    l1x -> array of pixel x coordinate
    l2x -> array of pixel y coordinate
    each pixel is surrounded by a gaussian with variance var
    """
    n = l1x.shape[0]
    l1x = l1x.reshape(n, 1)
    l2x = l2x.reshape(1, n)

    l1y = l1y.reshape(n, 1)
    l2y = l2y.reshape(1, n)

    coupling = np.exp(-((l1x - l2x) ** 2 +
                        (l1y - l2y) ** 2) / (4 * var))

    return np.einsum('i,j,ij->', p1, p2, coupling)


def snr(p1, l1x, l1y, p2, l2x, l2y, var):
    """
    Calculate the SNR between two images.

    Using the inner product defined above, calculates the SNR
        between two images given in sum of gaussian form
    See inner product for definitinos of variables
    Note the first set of pixels and pixel locations is
        considered to be the ground truth
    """
    ip12 = inner_product(p1, l1x, l1y, p2, l2x, l2y, var)
    ip11 = inner_product(p1, l1x, l1y, p1, l1x, l1y, var)
    ip22 = inner_product(p2, l2x, l2y, p2, l2x, l2y, var)

    return ip11 / (ip11 + ip22 - 2 * ip12)


class DataAnalyzer:
    """Class to analyze data from simulations."""

    def __init__(self, data):
        """
        Initialize the class.

        data - dictionary containing information about run
        Loads in data from file and saves parameters in class
        """
        self.data = data

        self.DT = self.data['DT']
        self.N_T = int(self.data['N_T'])

        self.xr = self.data['XR'][0]
        self.yr = self.data['YR'][0]
        self.S_gen = self.data['S_gen']
        self.Var = self.data['Var'][0]

        self.blur_sdev = float(np.sqrt(0.5))
        # float(np.sqrt(self.Var)) / self.data['ds']

        self.N_itr = self.data['N_itr']
        # self.DC_gen = self.data['DC_gen']
        # self.DC_infer = self.data['DC_infer']
        self.L_I = self.data['L_I']
        self.LAMBDA = self.data['lamb']
        self.R = self.data['R']
        self.L_N = self.data['L_N']

        # Convert retinal positions to grid
        xs = self.data['XS']
        ys = self.data['YS']

        xs, ys = np.meshgrid(xs, ys)
        self.xs = xs.ravel()
        self.ys = ys.ravel()

        self.N_itr = self.data['N_itr']
        self.var = self.data['Var'][0]

    @classmethod
    def fromfilename(cls, filename):
        """
        Initialize class from a file.

        Parameters
        ----------
        filename : str
            Path to file containing data file from EM run
                contains a datadict
        """
        data = pkl.load(open(filename, 'rb'))
        return cls(data)

    def snr_one_iteration(self, q):
        """
        Calculate the SNR of the estimated image and the true image.

        Parameters
        ----------
        q : int
            Iteration of the EM to pull estimated image.

        Note that we shift the image estimate by the average
            amount that the path estimate was off the true path
            (There is a degeneracy in the representation that this
            fixes. )
        """
        s_est = self.data['EM_data'][q]['image_est']
        t = self.data['EM_data'][q]['time_steps']

        try:
            xyr_est = self.data['EM_data'][q]['path_means']
            xr_est = xyr_est[:, 0]
            yr_est = xyr_est[:, 1]

            dx = np.mean(self.xr[0:t] - xr_est[0:t])
            dy = np.mean(self.yr[0:t] - yr_est[0:t])
        except KeyError:
            dx = 0.
            dy = 0.
        self.dx = dx
        self.dy = dy
        i1 = self.S_gen.ravel()
        i2 = s_est.ravel()
        i1 = i1 / i1.max()
        i2 = i2 / i2.max()
        return snr(i1, self.xs, self.ys,
                   i2, self.xs + dx, self.ys + dy,
                   self.var)

    def snr_list(self):
        """Return a list giving the SNR after each iteration."""
        return [self.snr_one_iteration(q) for q in range(self.N_itr)]

    def time_list(self):
        """Return a list of the times for each EM iteration in ms."""
        return (self.N_T * (np.arange(self.N_itr) + 1) /
                self.N_itr * 1000 * self.DT)

    def plot_path_estimate(self, q, d):
        """
        Plot the actual and estimated path generated.

        Parameters
        ----------
        q : int
            EM iteration number
        d : int
            Dimension to plot (either 0 or 1)
        """
        est_mean = self.data['EM_data'][q]['path_means']
        est_sdev = self.data['EM_data'][q]['path_sdevs']

        if (d == 0):
            path = self.xr
            label = 'Hor.'
            dxy = self.dx
        elif (d == 1):
            path = self.yr
            label = 'Ver.'
            dxy = self.dy
        else:
            raise ValueError('d must be either 0 or 1')

        # t = self.data['EM_data'][q]['time_steps']

        plt.fill_between(self.DT * np.arange(self.N_T),
                         est_mean[:, d] - est_sdev[:, d],
                         est_mean[:, d] + est_sdev[:, d],
                         alpha=0.5, linewidth=1.)
        plt.plot(self.DT * np.arange(self.N_T),
                 est_mean[:, d], label='estimate')
        plt.plot(self.DT * np.arange(self.N_T),
                 path, label='actual')
        plt.xlabel('Time (s)')
        plt.ylabel('Relative position (arcmin)')
        plt.title(label + ' Pos., shift = %.2f' % dxy)

    def plot_velocity_estimate(self, q, d):
        """
        Plot the estimate of the velocity by the EM algorithm.

        q - EM iteration number
        d - dimension to plot (0 or 1)
        """
        if not self.data['motion_prior']['mode'] == 'VelocityDiffusion':
            raise RuntimeError('No velocity for this motion prior')

        est_mean = self.data['EM_data'][q]['path_means']
        est_sdev = self.data['EM_data'][q]['path_sdevs']

        if d == 0:
            label = 'Hor.'
        elif d == 1:
            label = 'Ver.'
        else:
            raise ValueError('d must be either 0 or 1')

        # t = self.data['EM_data'][q]['time_steps']

        d = d + 2  # Correct index for est_mean

        plt.fill_between(self.DT * np.arange(self.N_T),
                         est_mean[:, d] - est_sdev[:, d],
                         est_mean[:, d] + est_sdev[:, d],
                         alpha=0.5, linewidth=1.)
        plt.plot(self.DT * np.arange(self.N_T),
                 est_mean[:, d], label=label + 'estimate')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (pixels/sec)')
        # plt.legend()

    def plot_dynamic_vars(self, q):
        """
        Plot all of the dynamic variables (x, y, vx, vy).

        q : int
            EM iteration to plot.
        """
        if not self.data['motion_prior']['mode'] == 'VelocityDiffusion':
            raise RuntimeError('Run has no velocity estimate')

        plt.subplot(2, 2, 1)
        self.plot_path_estimate(q, 0)

        plt.subplot(2, 2, 2)
        self.plot_path_estimate(q, 1)

        plt.subplot(2, 2, 3)
        self.plot_velocity_estimate(q, 0)

        plt.subplot(2, 2, 4)
        self.plot_velocity_estimate(q, 1)

    def plot_image_estimate(self, q):
        """Plot the estimated image after iteration q."""
        if q == -1:
            q = self.N_itr - 1

        res = _get_sum_gaussian_image(
            self.data['EM_data'][q]['image_est'].ravel(),
            self.xs, self.ys,
            self.data['ds'] / np.sqrt(2), n=100)
        plt.title('Estimated Image, S = DA:\n SNR = %.2f'
                  % self.snr_one_iteration(q))

        plt.imshow(res, cmap=plt.cm.gray, interpolation='nearest')
        plt.colorbar()

    def plot_base_image(self):
        """Plot the original image that generates the data."""
        res = _get_sum_gaussian_image(
            self.S_gen.ravel(), self.xs, self.ys,
            self.data['ds'] / np.sqrt(2), n=100)

        plt.title('Stationary Object in the World')
        plt.imshow(res, cmap=plt.cm.gray, interpolation='nearest')
        plt.colorbar()

    def plot_em_estimate(self, q):
        """Visualize the results after iteration q."""
        if q == -1:
            q = self.N_itr - 1

        n_time_steps = self.data['EM_data'][q]['time_steps']
        t_ms = self.DT * n_time_steps * 1000.

        fig = plt.figure(figsize=(12, 8))
        fig.suptitle('EM Reconstruction after t = {}'.format(t_ms))

        plt.subplot(2, 3, 2)
        self.plot_spikes(n_time_steps - 1, mode='ON')

        plt.subplot(2, 3, 3)
        self.plot_spikes(n_time_steps - 1, mode='OFF')

        plt.subplot(2, 3, 4)
        self.plot_image_estimate(q)

        plt.subplot(2, 3, 1)
        self.plot_base_image()

        plt.subplot(2, 3, 5)
        self.plot_path_estimate(q, 0)

        plt.subplot(2, 3, 6)
        self.plot_path_estimate(q, 1)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

    def save_images(self):
        """Save images for all iterations."""
        for q in range(self.N_itr):
            plt.clf()
            self.plot_EM_estimate(q)
            plt.savefig('img%d.png' % (100 + q))

    def compute_spike_moving_average(self, tau=0.005):
        """
        Compute the exponential moving average of the spikes.

        tau - time constant of moving average, should be a multiple of self.DT
        Saves an array self.Rav -> EMA of firing rate for each neuron
        """
        rho = 1 - self.DT / tau
        rav = np.zeros_like(self.R)

        rav[:, 0] = self.R[:, 0]
        for i in range(1, self.N_T):
            rav[:, i] = rho * rav[:, i - 1] + (1 - rho) * self.R[:, i]

        self.rav = rav / self.DT

    def plot_spikes(self, t, moving_average=True, mode='ON'):
        """
        Plot the spiking profile at timestep number t.

        t - timestep number to plot
        """
        if t > self.N_T:
            raise ValueError('time does not go past a certain time')
        if moving_average:
            try:
                self.rav
            except AttributeError:
                self.compute_spike_moving_average()
            s = self.rav[:, t]
        else:
            s = self.R[:, t]

        # FIXME: broken for hexagonal grid
        nn = self.L_N ** 2 * 2
        if mode == 'OFF':
            spikes = s[0: nn / 2]
        elif mode == 'ON':
            spikes = s[nn / 2: nn]
        else:
            raise ValueError('mode must be ON or OFF')

        vmin = 0.
        vmax = 200.

        if moving_average:
            st = 'Spike ExpMA'
        else:
            st = 'Spikes'

        plt.imshow(spikes.reshape(self.L_N, self.L_N),
                   interpolation='nearest', cmap=plt.cm.gray_r,
                   vmin=vmin, vmax=vmax)
        plt.title('{} for {} Cells at time {}'.format(st, mode, t))
#        plt.colorbar()

    def plot_firing_rates(self, t, mode='ON'):
        """
        Plot the firing rates for each neuron.

        Note: The visualization makes the most sense when the RFs of the
            neurons form a rectangular grid
        """
        frs = self.data['FP'][0] / self.DT
        nn = self.L_N ** 2 * 2
        if mode == 'OFF':
            fr = frs[0: nn / 2, t]
        elif mode == 'ON':
            fr = frs[nn / 2: nn, t]
        else:
            raise ValueError('mode must be ON or OFF')

        plt.imshow(fr.reshape(self.L_N, self.L_N),
                   interpolation='nearest',
                   cmap=plt.cm.gray,
                   vmin=0, vmax=100.)
        # t_str = ('lambda(t) (Hz) for {} Cells'.format(mode))
        # plt.title(t_str)

    def plot_fr_and_spikes(self, t):
        """
        Plot the base image, firing rate, and exponential moving.

            average of the spikes at time t.
        t : int
            Timestep to plot
        """
        plt.figure(figsize=(10, 8))

        plt.subplot(2, 2, 1)
        self.plot_base_image()

        plt.subplot(2, 2, 2)
        self.plot_firing_rates(t, mode='ON')
        plt.title('Retinal Image')

        # Spikes
        plt.subplot(2, 2, 3)
        self.plot_spikes(t, mode='ON', moving_average=True)

        plt.subplot(2, 2, 4)
        self.plot_spikes(t, mode='OFF', moving_average=True)

    def plot_rfs(self):
        """Create a plot of the receptive fields of the neurons."""
        self.xe = self.data['XE']
        self.ye = self.data['YE']
#        self.IE = self.data['IE']
        self.Var = self.data['Var']
        std = np.sqrt(np.mean(self.Var))
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_xlim((np.min(self.xe), np.max(self.xe)))
        ax.set_ylim((np.min(self.ye), np.max(self.ye)))
        for xe, ye in zip(self.xe, self.ye):
            circ = plt.Circle((xe, ye), std, color='b', alpha=0.4)
            fig.gca().add_artist(circ)

    def plot_tuning_curves(self, baseline_rate=10.):
        """Create a plot showing the tuning curves of the neurons."""
        x = np.arange(0, 1, 0.01)
        l0 = self.data['L0']
        l1 = self.data['L1']
        y_on = np.exp(np.log(l0) + x * np.log(l1 / l0))
        y_off = np.exp(np.log(l0) + (1 - x) * np.log(l1 / l0))
        plt.plot(x, y_on, label='ON')
        plt.plot(x, y_off, label='OFF')
        plt.plot(x, baseline_rate + 0 * x, '--')
        plt.xlabel('Stimulus intensity')
        plt.ylabel('Firing Rate (Hz)')
        plt.title('Firing rate as a function of Stimulus Intensity')
        plt.legend()

    def save_em_jpgs(self, output_dir, tag):
        """
        Save the figures from plot_EM_estimate for all iterations.

        output_dir : str
            String for the output directory
        tag : str
            String describing the run
        """
        for i in range(self.N_itr):
            self.plot_EM_estimate(i)
            plt.savefig(os.path.join(
                output_dir,
                'em_est_{}_{:03}.jpg'.format(tag, i)), dpi=50)

    def plot_image_and_rfs(self, s=150):
        """Plot the image with the neuron RF centers."""
        _plot_image_and_rfs(
            self.data['XE'], self.data['YE'], self.data['de'],
            self.xs, self.ys, self.data['ds'],
            self.xr, self.yr,
            self.data['S_gen'], s)


def _plot_image_and_rfs(xe, ye, de, xs, ys, ds, xr, yr, s_gen,
                        s=150):
    """
    Plot the image and the receptive fields.

    xe, ye: array, shape (n_n,)
        Neuron RF centers
    xs, ys: array, shape (l_i ** 2,)
        Locations of pixel centers
    xr, yr: array, shape (1, n_t)
        Location of eye at time t
    de, ds : float
        Neuron, pixel spacing
    S_gen : array, shape (l_i ** 2,)
        Values of pixels
    s : float
        Size of markers for pixels in pix ** 2
    """
    plt.scatter(xe, ye,
                label='Neuron RF Centers, de={}'.format(de),
                alpha=1.0)
    plt.scatter(xs, -ys, c=s_gen.ravel(), cmap=plt.cm.gray_r,
                label='Pixel Centers, ds={}'.format(ds), s=s, alpha=0.5)
    plt.axes().set_aspect('equal')
    if xr is not None and yr is not None:
        plt.plot(xr[::5], yr[::5], label='Eye path', c='g')
    plt.legend()


def _get_sum_gaussian_image(s_gen, xs, ys, sdev, n=50):
    """
    Plot a sum of Gaussians with given weights and centers.

    Parameters
    ----------
    s_gen : array, shape (n_pix,)
        Values of pixels.
    xs, ys : array, shape (n_pix,)
        X and Y locations of the pixels.
    sdev : float
        Standard deviation of the Gaussians
    n : int
        Number of samples to get for sum of Gaussians

    Returns
    -------
    res : float array, shape (n, n)
        Image of the sum of Gaussians
    """
    m1, m2 = xs.min(), xs.max()
    xx = np.linspace(m1, m2, n)
    XX, YY = np.meshgrid(xx, xx)
    XX, YY = [u.ravel()[np.newaxis, :] for u in [XX, YY]]
    xs, ys, S_gen = [u[:, np.newaxis] for u in [xs, ys, s_gen]]
    res = np.sum(
        S_gen * np.exp(((xs - XX) ** 2 + (ys - YY) ** 2) /
                       (-2 * sdev ** 2)), axis=0)
    return res.reshape(n, n)


def pf_plot(pf, t):
    """
    Plot the particles and associated weights of the particle filter.

    at a certain time. Weight is proportional to the area of the marker.

    pf : Particle Filter
        Particle filter class
    t : int
        Time point to plot the particles and weights
    """
    xx = pf.XS[t, :, 0]
    yy = pf.XS[t, :, 1]
    ww = pf.WS[t, :]
    plt.scatter(xx, yy, s=ww * 5000)


def plot_fill_between(t, data, label='', c=None, k=1.):
    """
    Create a plot of the data +/- k standard deviations.

    Parameters
    ----------
    t : array, shape (timesteps, )
        Times for each data point
    data : array, shape (samples, timesteps)
        Data to plot mean and +/- one sdev as a function of time
    k : float
        Scaling factor for standard deviations
    """
    mm = data.mean(0)
    sd = data.std(0) * k
    plt.fill_between(t, mm - sd, mm + sd, alpha=0.5, color=c)
    plt.plot(t, mm, color=c, label=label)


if __name__ == '__main__':
    fn = sys.argv[1]
    da = DataAnalyzer.fromfilename(fn)
