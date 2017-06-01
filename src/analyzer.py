"""Code to analyze the output of the simulations."""

import numpy as np
import cPickle as pkl
import sys
import os
# from scipy.ndimage.filters import gaussian_filter
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from utils.h5py_utils import LazyDict

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
    l1x = l1x.reshape(n, 1).astype('float32')
    l2x = l2x.reshape(1, n).astype('float32')

    l1y = l1y.reshape(n, 1).astype('float32')
    l2y = l2y.reshape(1, n).astype('float32')
    var = var.astype('float32')

    coupling = np.exp(-((l1x - l2x) ** 2 +
                        (l1y - l2y) ** 2) / (4 * var))

    #  return np.einsum('i,j,ij->', p1, p2, coupling)
    return np.dot(p1, coupling).dot(p2)

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

        self.xr = self.data['XR'][0].astype('float32')
        self.yr = self.data['YR'][0].astype('float32')
        self.S_gen = self.data['S_gen'].astype('float32')
        self.Var = self.data['Var'][0].astype('float32')

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
        xs = self.data['XS'].astype('float32')
        ys = self.data['YS'].astype('float32')


        xs, ys = np.meshgrid(xs, ys)
        self.xs = xs.ravel()
        self.ys = ys.ravel()

        self.N_itr = self.data['N_itr']

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
        data = LazyDict(fn=filename)
        return cls(data)

    def snr_one_iteration(self, q, img=None):
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
        snr_key = 'EM_data/{}/SNR'.format(q)
        try:
            snr1 = self.data[snr_key]
        except KeyError:
            #  print 'Computing SNR for iteration: {}'.format(q)

            q_data = self.data['EM_data/{}'.format(q)]

            s_est = q_data['image_est']
            t = q_data['time_steps']

            try:
                xyr_est = q_data['path_means']
                xr_est = xyr_est[:, 0]
                yr_est = xyr_est[:, 1]

                dx = np.mean(self.xr[0:t] - xr_est[0:t])
                dy = np.mean(self.yr[0:t] - yr_est[0:t])
            except KeyError:
                dx = 0.
                dy = 0.
            self.dx = dx
            self.dy = dy
            if img is None:
                img = self.S_gen
            i1 = img.ravel()
            i2 = s_est.ravel()
            i1 = i1 / i1.max()
            i2 = i2 / i2.max()
            snr1 = snr(i1, self.xs, self.ys,
                       i2, self.xs + dx, self.ys + dy,
                       var=(self.data['ds'] * 0.5) ** 2)
            self.data[snr_key] = snr1

        return snr1

    def snr_list(self):
        """Return a list giving the SNR after each iteration."""
        return [self.snr_one_iteration(q) for q in range(self.N_itr)]

    def time_list(self):
        """Return a list of the times for each EM iteration in ms."""
        return (self.N_T * (np.arange(self.N_itr) + 1) /
                self.N_itr * 1000 * self.DT)

    def plot_path_estimate(self, ax, q, d, title=True):
        """
        Plot the actual and estimated path generated.

        Parameters
        ----------
        q : int
            EM iteration number
        d : int
            Dimension to plot (either 0 or 1)
        """
        q_data = self.data['EM_data/{}'.format(q)]
        est_mean = q_data['path_means']
        est_sdev = q_data['path_sdevs']

        if (d == 0):
            path = self.xr
            label = 'X'
            #  dxy = self.dx
        elif (d == 1):
            path = self.yr
            label = 'Y'
            #  dxy = self.dy
        else:
            raise ValueError('d must be either 0 or 1')


        ax.fill_between(self.DT * np.arange(self.N_T),
                        est_mean[:, d] - est_sdev[:, d],
                        est_mean[:, d] + est_sdev[:, d],
                        alpha=0.5, linewidth=0.25)
        ax.plot(self.DT * np.arange(self.N_T),
                 est_mean[:, d], label='estimate')
        ax.plot(self.DT * np.arange(self.N_T),
                 path, label='actual')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Relative position (arcmin)')
        if title:
            #  ax.set_title(label + ' Pos., shift = %.2f' % dxy)
            ax.set_title('{} Position'.format(label))

    #  def plot_velocity_estimate(self, q, d):
    #      """
    #      Plot the estimate of the velocity by the EM algorithm.

    #      q - EM iteration number
    #      d - dimension to plot (0 or 1)
    #      """
    #      if not self.data['motion_prior/mode'] == 'VelocityDiffusion':
    #          raise RuntimeError('No velocity for this motion prior')

    #      est_mean = self.data['EM_data'][q]['path_means']
    #      est_sdev = self.data['EM_data'][q]['path_sdevs']

    #      if d == 0:
    #          label = 'Hor.'
    #      elif d == 1:
    #          label = 'Ver.'
    #      else:
    #          raise ValueError('d must be either 0 or 1')

    #      # t = self.data['EM_data'][q]['time_steps']

    #      d = d + 2  # Correct index for est_mean

    #      plt.fill_between(self.DT * np.arange(self.N_T),
    #                       est_mean[:, d] - est_sdev[:, d],
    #                       est_mean[:, d] + est_sdev[:, d],
    #                       alpha=0.5, linewidth=1.)
    #      plt.plot(self.DT * np.arange(self.N_T),
    #               est_mean[:, d], label=label + 'estimate')
    #      plt.xlabel('Time (s)')
    #      plt.ylabel('Velocity (pixels/sec)')
    #      # plt.legend()

    #  def plot_dynamic_vars(self, q):
    #      """
    #      Plot all of the dynamic variables (x, y, vx, vy).

    #      q : int
    #          EM iteration to plot.
    #      """
    #      if not self.data['motion_prior']['mode'] == 'VelocityDiffusion':
    #          raise RuntimeError('Run has no velocity estimate')

    #      plt.subplot(2, 2, 1)
    #      self.plot_path_estimate(q, 0, title=False)

    #      plt.subplot(2, 2, 2)
    #      self.plot_path_estimate(q, 1, title=False)

    #      plt.subplot(2, 2, 3)
    #      self.plot_velocity_estimate(q, 0)

    #      plt.subplot(2, 2, 4)
    #      self.plot_velocity_estimate(q, 1)

    def plot_image_estimate(self, fig, ax, q, cmap=plt.cm.gray,
                            colorbar=True, vmax=None):
        """Plot the estimated image after iteration q."""
        if q == -1:
            q = self.N_itr - 1


        image_est = self.data['EM_data/{}/image_est'.format(q)].ravel()
        res = _get_sum_gaussian_image(
            image_est,
            self.xs, self.ys,
            self.data['ds'] / np.sqrt(2), n=100)
        ax.set_title('Estimated Image, S = DA:\n SNR = %.2f'
                  % self.snr_one_iteration(q))
        # FIXME: extent calculation could break in future
        a = self.data['ds'] * self.L_I / 2
        cax = ax.imshow(res, cmap=cmap, interpolation='nearest',
                        extent=[-a, a, -a, a],
                        vmax=vmax)
        if colorbar:
            fig.colorbar(cax, ax=ax)

    def plot_base_image(self, fig, ax, colorbar=True, alpha=1.,
                        cmap=plt.cm.gray_r, dx=0., dy=0.):
        """Plot the original image that generates the data."""
        res = _get_sum_gaussian_image(
            self.S_gen.ravel(), self.xs, self.ys,
            self.data['ds'] / np.sqrt(2), n=100)
        a = self.data['ds'] * self.L_I / 2
        cax = ax.imshow(res, cmap=cmap, interpolation='nearest',
                        extent=[-a - dx, a - dx, -a + dy, a + dy],
                        alpha=alpha)
        if colorbar:
            fig.colorbar(cax, ax=ax)
            ax.set_title('Stationary Object in the World')

    def plot_em_estimate(self, q, figsize=None):
        """Visualize the results after iteration q."""
        if q == -1:
            q = self.N_itr - 1

        n_time_steps = self.data['EM_data/{}/time_steps'.format(q)]
        t_ms = self.DT * n_time_steps * 1000.

        if figsize is None:
            figsize=(3.5, 5)
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=figsize)
        #  fig.suptitle('EM Reconstruction after t = {} ms'.format(t_ms))

        self.plot_spikes(ax[1][0], n_time_steps - 1, mode='ON')

        self.plot_spikes(ax[1][1], n_time_steps - 1, mode='OFF')

        self.plot_image_estimate(fig, ax[0][1], q, cmap=plt.cm.gray_r)

        self.plot_image_and_rfs(fig, ax=ax[0][0], legend=False, q=n_time_steps - 1)
        ax[0][0].set_title('Pattern with Cone RFs')

        self.plot_path_estimate(ax[2][0], q, 0)

        self.plot_path_estimate(ax[2][1], q, 1)

        for u in [0, 1]:
            ax_ = ax[2][u]
            start, stop = ax_.get_xlim()
            ticks = np.arange(start, stop + 0.1, (stop - start) / 2)
            ax_.set_xticks(ticks)

        u0, v0 = ax[2][0].get_ylim()
        u1, v1 = ax[2][1].get_ylim()

        for u in [0, 1]:
            ax[2][u].set_ylim(min(u0, u1), max(v0, v1))





        plt.tight_layout()
        #  plt.subplots_adjust(top=0.9)

        return fig, ax

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

        rav[:, 0] = self.R[:, 0] * (1 - rho)
        for i in range(1, self.N_T):
            rav[:, i] = rho * rav[:, i - 1] + (1 - rho) * self.R[:, i]

        self.rav = rav / self.DT

    def plot_spikes(self, ax, t, moving_average=True, mode='ON'):
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

        xe, ye, ie = self.data['XE'], self.data['YE'], self.data['IE']
        de = self.data['de']
        ON, OFF = 0, 1

        def normalize(s0, mm=1.5 * self.data['L1']):
            if s0 < mm:
                return s0 / (1.0 * mm)
            else:
                return 1.

        for x, y, i, s0 in zip(xe, ye, ie, s):
            if i == ON  and mode == 'OFF':
                continue
            if i == OFF and mode == 'ON':
                continue
            alpha = normalize(s0)
            ax.add_patch(plt.Circle((x, -y), de * 0.203, alpha=alpha))
        if moving_average:
            st = 'ExpMA'
        else:
            st = 'Spikes'
        ax.set_title('{} for {} Cells'.format(st, mode))
        m = max(max(xe), max(ye))
        ax.set_xlim([-m, m])
        ax.set_ylim([-m, m])
        ax.set_aspect('equal')

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
        ax = plt.subplot(2, 2, 3)
        self.plot_spikes(ax, t, mode='ON', moving_average=True)

        ax = plt.subplot(2, 2, 4)
        self.plot_spikes(ax, t, mode='OFF', moving_average=True)

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
        x = np.arange(0, 1 + 0.01, 0.01)
        l0 = self.data['L0']
        l1 = self.data['L1']
        y_on = np.exp(np.log(l0) + x * np.log(l1 / l0))
        y_off = np.exp(np.log(l0) + (1 - x) * np.log(l1 / l0))
        plt.plot(x, y_on, label='ON')
        plt.plot(x, y_off, label='OFF')
        plt.plot(x, baseline_rate + 0 * x, '--')
        plt.xlabel('Stimulus intensity')
        plt.ylabel('Firing Rate (Hz)')
        #  plt.title('Firing rate as a function \n of Stimulus Intensity')
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

    def plot_image_and_rfs(self, fig=None, ax=None, legend=True, q=None, alpha_rf=0.5):
        """Plot the image with the neuron RF centers."""
        if fig is None:
            fig, ax = plt.subplots(1, 1)

        if q is None:
            dx, dy = 0., 0.
            xr, yr = self.xr, self.yr

        else:
            dx = self.xr[q]
            dy = self.yr[q]
            xr, yr = self.xr[0:q], self.yr[0:q]

        self.plot_base_image(fig, ax, colorbar=False, alpha=1.,
                             cmap=plt.cm.gray_r,
                             dx=dx, dy=dy)


        _plot_rfs(
            ax, self.data['XE'], self.data['YE'], self.data['de'],
            legend, alpha=alpha_rf)

        ax.plot(-xr, yr, label='Eye path', c='g')

        m = max(max(self.data['XE']), max(self.data['YE']))
        ax.set_xlim([-m, m])
        ax.set_ylim([-m, m])


    def plot_moving_image_and_spikes(self, q):
        plt.figure(figsize=(12, 4))
        ax = plt.subplot(1, 3, 1)
        self.plot_image_and_rfs(ax, q=q, legend=False)
        ax = plt.subplot(1, 3, 2)
        self.plot_spikes(ax, q, mode='ON')
        ax = plt.subplot(1, 3, 3)
        self.plot_spikes(ax, q, mode='OFF')




def _plot_rfs(ax, xe, ye, de, legend, alpha=0.5):
    """
    Plot the image and the receptive fields.

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
    r = 0.203 * de
    for i, (x, y) in enumerate(zip(xe, ye)):
        if i == 0:
            label = None  # 'One SDev of Neuron RF'
        else:
            label = None
        ax.add_patch(plt.Circle((x, -y), r, color='red', fill=True, alpha=alpha,
                                label=label))
    #  plt.scatter(xe, ye, alpha=0.5, label='Neuron RF Centers, de={}'.format(de))

    if legend:
        plt.legend()
        ax.set_xlabel('x (arcmin)')
        ax.set_ylabel('y (arcmin)')

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


def plot_fill_between(ax, t, data, label='', c=None, hatch=None, k=1.):
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
    ax.fill_between(t, mm - sd, mm + sd, alpha=0.5, color=c,
                    hatch=hatch)
    ax.plot(t, mm, color=c, label=label)


if __name__ == '__main__':
    fn = sys.argv[1]
    da = DataAnalyzer.fromfilename(fn)
