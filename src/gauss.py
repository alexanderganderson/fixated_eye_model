"""Implement Gaussian Model."""

import numpy as np
import tensorflow as tf
import utils.particle_filter as pf
from utils.particle_filter import LikelihoodPotential
from utils.hex_lattice import gen_hex_lattice
from utils.path_generator import (DiffusionPathGenerator,
                                  ExperimentalPathGenerator)


class EMGauss(object):
    """Produce spikes and infer underlying causes."""

    def __init__(self,
                 l_i,
                 motion_gen,
                 motion_prior,
                 n_t,
                 ds,
                 de,
                 n_p,
                 print_mode,
                 l_n,
                 sig_obs,
                 rf_ratio=0.203,
                 dt=0.001,
                 neuron_layout='hex'):

        xs, ys, n_pix = self.init_pix_centers(l_i=l_i, ds=ds)
        xe, ye, n_n = self.init_rf_centers(mode=neuron_layout, l_n=l_n, de=de)

        # FIMXE: make i,j tensor
        var_s = np.ones((1,), dtype='float32') * (
            (0.5 * ds) ** 2 + (rf_ratio * de) ** 2)
        var_m = np.ones((n_n,), dtype='float32') * sig_obs ** 2

        self.pg = self.init_path_generator(motion_gen, n_t=n_t, dt=dt)
        self.tb = TFBackend(xs, ys, xe, ye, var_s, var_m, n_t=n_t, n_p=n_p)
        self.pf = self.init_particle_filter(
            motion_prior=motion_prior, n_p=n_p, dt=dt, n_n=n_n, n_t=n_t)

        self.data = {}
        self.n_pix = n_pix
        self.print_mode = print_mode

    @staticmethod
    def init_path_generator(motion_gen, n_t, dt):
        """
        Initialize path generator.

        Parameters
        ----------
        motion_gen : dict
            Gives options for Path generators.
        n_t : int
            Number of timesteps in path.
        dt : float
            Timestep

        Returns
        -------
        pg : PathGenerator
            Object that generates paths.
        """
        if motion_gen['mode'] == 'Diffusion':
            return DiffusionPathGenerator(
                n_t=n_t, lx=100, dc=motion_gen['dc'], dt=dt)
        elif motion_gen['mode'] == 'Experiment':
            return ExperimentalPathGenerator(
                n_t, motion_gen['fpath'], dt)
        else:
            raise ValueError(
                'motion_gen[mode] must be Diffusion of Experiment')

    @staticmethod
    def init_pix_centers(l_i, ds):
        """
        Initialize the centers of the pixels.

        Parameters
        ----------
        l_i : int
            Length of the image in pixels.
        ds : float
            Spacing of pixels.

        Returns
        -------
        xs, ys : array, shape (l_i,)
            X, Y coordinates of the pixels.
        """
        tmp = (np.arange(l_i) - (l_i - 1) / 2.) * ds
        xs, ys = np.meshgrid(tmp, tmp)
        xs, ys = [u.astype('float32').ravel() for u in [xs, ys]]
        n_pix = xs.size
        return xs, ys, n_pix

    @staticmethod
    def init_rf_centers(mode, l_n, de):
        """
        Initialize the receptive field centers.

        Parameters
        ----------
        mode : str
            Type of lattice. Either sqr or hex.
        l_n : int
            Length of lattice.
        de : float
            Spacing of RFs

        Returns
        -------
        xe, ye : array, shape (n_sensor,)
            X, Y positions of sensors.
        n_n : int
            Number of sensors
        """
        if mode == 'sqr':
            xe, ye = np.meshgrid(
                de * np.arange(- l_n / 2, l_n / 2),
                de * np.arange(- l_n / 2, l_n / 2))
            xe, ye = xe.ravel(), ye.ravel()
        elif mode == 'hex':
            xe, ye = gen_hex_lattice(l_n * de, a=de)
            n_n = xe.size
            xe, ye = [xy + (np.random.rand(n_n) - 0.5) * de * 0.25
                      for xy in [xe, ye]]
        else:
            raise ValueError('Unrecognized Neuron Mode {}'.format(mode))
        xe, ye = xe.astype('float32'), ye.astype('float32')
        n_n = xe.size
        return xe, ye, n_n

    def gen_data(self, s_gen, pg=None):
        """
        Generate data given an image.

        Parameters
        ----------
        s_gen : array, shape (n_pix,)
            Image to generate data from.
        pg : PathGenerator
            Object to generate path. If None, use default.

        Returns
        -------
        m : array, shape (n_sensors, n_t)
            Measurements.
        """
        if pg is None:
            pg = self.pg
        path = pg.gen_path()
        xr = path[0]
        yr = path[1]
        m = self.tb.get_m(path[0], path[1], s_gen)
        return m, xr, yr

    @staticmethod
    def init_particle_filter(motion_prior, n_p, dt, n_n, n_t):
        """
        Initialize particle filter.

        Parameters
        ----------
        motion_prior : dict
            Contains diffusion constant.
        n_p : int
            Number of particles to use.
        dt : float
            Timestep in seconds.
        n_n : int
            Number of sensors.

        Returns
        -------
        pf : ParticleFilter
            Particle Filter object.
        """
        # Diffusion Prior
        dc_infer = motion_prior['dc']
        d_h = 2  # Dimension of hidden state (i.e. x,y = 2 dims)
        d_o = n_n
        sdev = np.sqrt(dc_infer * dt / 2) * np.ones((d_h,))
        ipd = pf.GaussIPD(d_h, d_o, sdev * 0.001)
        tpd = pf.GaussTPD(d_h, d_o, sdev)
        ip = pf.GaussIP(d_h, sdev * 0.001)
        tp = pf.GaussTP(d_h, sdev)
        lp = GaussObsLP(d_h, n_n)
        y = np.zeros((n_t, 1))
        res = pf.ParticleFilter(
            ipd=ipd, tpd=tpd, ip=ip, tp=tp, lp=lp, Y=y, N_P=n_p)
        return res

    def reset(self):
        """Reset the simulation."""
        self.pf.reset()
        self.pf.lp.energy_func = None
        self.tb.reset()
        self.data = {}

    def _run_e(self, m, t_0, t_f, s):
        """
        Run a partial e step.

        Parameters
        ----------
        m : array, shape (n_sensors, n_t)
        t_0, t_f : int
            Initial and final times.
        s : array, shape (n_pix,)
            Current image estimate.

        Returns
        -------
        xr, yr : array, shape (n_particles, n_t)
            X, Y positions for each particle
        w0 : array, shape (n_particles,)
            Weights corresponding to each position.
        """
        def _energy_func(xr, yr, m):
            return self.tb.get_e(xr, yr, m, s)
        self.pf.lp.energy_func = _energy_func
        for t in range(t_0, t_f):
            assert self.pf.t == t
            self.pf.advance(Y=m[:, t-t_0])

        xr = self.pf.XS[t_0:t_f, :, 0].transpose()
        yr = self.pf.XS[t_0:t_f, :, 1].transpose()
        w0 = self.pf.WS[t_0:t_f, :].transpose()
        return xr, yr, w0

    def _run_m(self, a, b, xr, yr, w, m):
        """
        Run a partial m step.

        Parameters
        ----------
        a : array, shape (n_pix, n_pix)
            Current estimate of image inverse covariance.
        b : array, shape (n_pix,)
            Current estimate of bias.
        xr, yr : array, shape (n_p, n_t)
            Relative position of eye and image for particles.
        w : array, shape (n_p, n_t)
            Associated weights.
        m : array, shape (n_sensors, n_t)
            Measurements.

        Returns
        -------
        a : array, shape (n_pix, n_pix)
            Updated estimate of image inverse covariance.
        b : array, shape (n_pix,)
            Updated bias estimate.
        s : array, shape (n_pix,)
            New image estimate.
        """
        a0, b0 = self.tb.get_ab(xr=xr, yr=yr, w=w, m=m)
        a = a + a0
        b = b + b0
        s = np.linalg.solve(a, b)
        return a, b, s

    def run_em(self, m, n_passes, n_itr, reg=0.01):
        """
        Run EM algorithm to recover the image.

        Parameters
        ----------
        m : array, shape (n_sensors, n_t)
            Measurements.
        n_passes : int
            Number of full EM iterations.
        n_itr : int
            Number of pieces to break EM into.
        reg : float
            Regularization constant.

        Returns
        -------
        s : array, shape (n_pix,)
            Final estimate of the image.
        data : list of array, shape (n_pix,)
            List of estimated image at each step
        """
        n_pix = self.n_pix
        n_t = m.shape[-1]
        a = np.zeros((n_pix, n_pix))
        a += reg * np.eye(n_pix)
        b = np.zeros((n_pix,))
        s = np.zeros((n_pix,))

        data = []
        for _ in range(n_passes):
            self.pf.reset()
            #  a[:] = 0
            #  b[:] = 0
            for u in range(n_itr):
                t_0 = n_t * u / n_itr
                t_f = n_t * (u + 1) / n_itr

                m0 = m[:, t_0:t_f]
                xr0, yr0, w0 = self._run_e(m=m0, t_0=t_0, t_f=t_f, s=s)
                a, b, s = self._run_m(a=a, b=b, xr=xr0, yr=yr0, w=w0, m=m0)
                data.append(s.copy())
        return s, data

    def _build_param_dict(self):
        pass

    def _save(self):
        pass


def _get_t_matrix(t_xs, t_ys, t_xe, t_ye, t_xr, t_yr, t_var):
    """
    Get the 'T' matrix connecting pixels and receptive fields.

    Parameters
    ----------
    t_xs, t_ys : tf.Variable, shape (n_pix,)
        X, Y coordinates of image pixels.
    t_xe, t_ye : tf.Variable, shape (n_sensors,)
        X, Y coordinates of image sensors.
    t_xr, t_yr : tf.Variable, shape (n_p, n_t)
        X, Y coordinates of translations
    t_var : tf.Variable, shape (1,)
        Combined variances of pixel blurring and measuremnt RF size.

    Returns
    -------
    t_t, tf.Variable, shape (n_pix, n_sensors, n_translations)
        Matrix connecting pixels and samples.
    """

    c = tf.constant(1, dtype='float32', shape=(1,))
    t_xs, t_ys = [tf.einsum('i,j,p,t->ijpt', u, c, c, c) for u in [t_xs, t_ys]]

    t_xe, t_ye = [tf.einsum('i,j,p,t->ijpt', c, u, c, c) for u in [t_xe, t_ye]]

    t_xr, t_yr = [tf.einsum('i,j,pt->ijpt', c, c, u) for u in [t_xr, t_yr]]

    t_d2 = (t_xs - t_xe - t_xr) ** 2 + (t_ys - t_ye - t_yr) ** 2
    PI = tf.constant(np.pi, dtype='float32')
    t_t = tf.exp(- t_d2 / (2 * t_var)) / (2 * PI * t_var)
    return t_t


def _calc_a(t_w, t_t):
    """
    Get A matrix

    Parameters
    ----------
    t_w : tf.Tensor, shape (n_p, n_t)
        Particle filter weights
    t_t : tf.Tensor, shape (n_pix, n_sensors, n_p, n_t)
        Tensor connecting pixels and samples

    Returns
    -------
    t_a : tf.Tensor, shape (n_pix, n_pix)
        Inverse covariance of image estimate.
    """
    return tf.einsum('pt,xjpt,yjpt->xy', t_w, t_t, t_t)


def _calc_b(t_w, t_m, t_t):
    """
    Get B vector

    Parameters
    ----------
    t_w : tf.Tensor, shape (n_p, n_t)
        Particle filter weights
    t_m : tf.Tensor, shape (n_receptors, n_t)
        Measurements
    t_t : tf.Tensor, shape (n_pix, n_sensors, n_p, n_t)

    Returns
    -------
    t_b : tf.Tensor, shape (n_pix,)

    """
    return tf.einsum('pt,jt,ijpt->i', t_w, t_m, t_t)


def _calc_batched_e(t_m, t_s, t_t, t_var_m):
    """
    Calculate log p(M|X, S) batched over X.

    Parameters
    ----------
    t_m : tf.Tensor, shape (n_sensors, n_t)
        Measurements
    t_s : tf.Tensor, shape (n_pixels,)
        Pixels of image estimate
    t_t : tf.Tensor, shape (n_pixels, n_sensors, n_particles, n_t)
        Image to measurement matrix
    t_var_m : tf.Tensor, shape (n_sensors,)
        Variance of each measurement device

    Returns
    -------
    t_e : tf.Tensor, shape (n_particles, n_t)
        Energy batched over positions.
    """
    c = tf.constant(1, dtype='float32', shape=(1,))
    t_s = tf.einsum('i,j,p,t->ijpt', t_s, c, c, c)
    #  t_m0 = tf.einsum('ijpt,ijpt->jpt', t_s, t_t)
    t_m0 = tf.reduce_sum(t_s * t_t, axis=0)
    t_m = tf.einsum('p,jt->jpt', c, t_m)
    #  t_e = tf.einsum('jpt,j->pt', (t_m - t_m0) ** 2, 1./(2 * t_var_m))
    t_e = tf.reduce_sum((t_m - t_m0) ** 2 * 1./(2 * t_var_m), axis=0)
    return t_e


def _calc_m_gen(t_s, t_t, t_var_m, t_eps):
    """
    Get the measurements.

    Parameters
    ----------
    t_s : tf.Tensor, shape (n_pix,)
        Image that generates the measurements.
    t_t : tf.Tensor, shape (n_pix, n_sensors, n_particles, n_t)
        Image to measurement matrix.
    t_var_m : tf.Tensor, shape (n_sensors,)
        Variances of measurements
    t_eps : tf.Tensor, shape (n_sensors, n_p, n_t)

    Returns
    -------
    t_m : tf.Tensor, shape (n_sensors, n_p, n_t)
        Measurements
    """
    c = tf.constant(1, dtype='float32', shape=(1,))

    t_sp = tf.einsum('i,j,p,t->ijpt', t_s, c, c, c)
    #  import pdb; pdb.set_trace()
    #  t_m0 = tf.einsum('ijpt,ijpt->jpt', t_sp, t_t)
    t_m0 = tf.reduce_sum(t_sp * t_t, axis=0)

    t_sig_m = t_var_m ** 0.5
    t_sig_m = tf.einsum('j,p,t->jpt', t_sig_m, c, c)

    t_m = t_m0 + t_sig_m * t_eps
    return t_m


class TFBackend(object):
    """Runs core operations on GPU."""

    def __init__(self, xs, ys, xe, ye, var_s, var_m, n_t, n_p, SMIN=0., SMAX=1.):
        """
        Initialize backend that does core computations on GPU.

        Parameters
        ----------
        xs, ys : array, shape (n_pix,)
            X, Y coordinates of pixel centers.
        xe, ye : array, shape (n_sensors,)
            X, Y coordinates of receptor centers.
        var_s : array, shape (n_sensors,)
            Spatial variance of receptors (combines image smoothing and
                receptor spatial variance)
        var_m : array, shape (n_sensors,)
            Variance of each receptor.
        SMIN, SMAX : float
            Minimium, Maximum value for image pixels.
        """
        # Simulation constants
        t_xs, t_ys, t_xe, t_ye, t_var_s, t_var_m = [
            tf.Variable(u, name=name, dtype='float32') for u, name in
            zip([xs, ys, xe, ye, var_s, var_m],
                ['xs', 'ys', 'xe', 'ye', 'var_s', 'var_m'])]

        n_pix = xs.size
        n_sensors = var_m.shape[0]

        #########################
        # Generate measurements #
        #########################
        t_xr_gen, t_yr_gen = [tf.placeholder('float32', shape=(1, n_t), name=name)
                              for name in ['xr', 'yr']]  # p, t

        # i, j, b, t
        t_t_gen = _get_t_matrix(t_xs=t_xs, t_ys=t_ys,
                                t_xe=t_xe, t_ye=t_ye,
                                t_xr=t_xr_gen, t_yr=t_yr_gen,
                                t_var=t_var_s)


        t_s_gen = tf.placeholder('float32', shape=(n_pix,), name='s_gen')

        # j, p, t
        t_eps = tf.placeholder('float32', shape=(n_sensors, 1, None),
                               name='eps')
        t_m_gen = _calc_m_gen(t_s_gen, t_t_gen, t_var_m, t_eps)

        ################################
        # Generate inference equations #
        ################################
        t_xr, t_yr = [tf.placeholder('float32', shape=(n_p, None), name=name)
                      for name in ['xr', 'yr']]  # p, t

        # i, j, b, t
        t_t = _get_t_matrix(t_xs=t_xs, t_ys=t_ys,
                            t_xe=t_xe, t_ye=t_ye,
                            t_xr=t_xr, t_yr=t_yr,
                            t_var=t_var_s)


        t_w = tf.placeholder('float32', shape=(n_p, None), name='w')  # p, t
        t_m = tf.placeholder('float32', shape=(n_sensors, None), name='m')  # j, t
        t_s_inf = tf.placeholder('float32', shape=(n_pix,), name='s_inf')

        t_a = _calc_a(t_w, t_t)
        t_b = _calc_b(t_w, t_m, t_t)

        # Batched energy for particle filter
        t_e = _calc_batched_e(t_m, t_s_inf, t_t, t_var_m)

        # Initialize variables
        init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        with self.sess.as_default():
            self.sess.run(init_op)

        self.t_xs = t_xs
        self.t_ys = t_ys
        self.t_xe = t_xe
        self.t_ye = t_ye
        self.t_var_s = t_var_s
        self.t_var_m = t_var_m
        self.t_xr_gen = t_xr_gen
        self.t_yr_gen = t_yr_gen
        self.t_t_gen = t_t_gen
        self.t_s_gen = t_s_gen
        self.t_eps = t_eps
        self.t_m_gen = t_m_gen

        self.t_xr = t_xr
        self.t_yr = t_yr
        self.t_t = t_t
        self.t_w = t_w
        self.t_m = t_m
        self.t_s_inf = t_s_inf
        self.t_a = t_a
        self.t_b = t_b
        self.t_e = t_e

    def reset(self):
        """Reset the backend."""
        pass

    def get_m(self, xr, yr, s_gen):
        """
        Get measurements given a set of positions.

        Parameters
        ----------
        xr, yr : array, shape (n_t,)
            X, Y positions of views of the image.
        s_gen : array, shape (n_pix,)
            Image to generate measurements from.

        Returns
        -------
        m : array, shape (n_sensors, n_t)
            Measurements
        """
        n_t = xr.shape[0]
        n_sensors = self.t_eps.get_shape()[0]
        noise = np.random.randn(n_sensors, 1, n_t)
        noise = np.clip(noise, -3, 3)
        feed_dict = {
            self.t_xr_gen: xr[np.newaxis, :],
            self.t_yr_gen: yr[np.newaxis, :],
            self.t_s_gen: s_gen,
            self.t_eps: noise
        }
        with self.sess.as_default():
            m = self.sess.run(self.t_m_gen, feed_dict=feed_dict)
        return m[:, 0, :]

    def get_ab(self, xr, yr, w, m):
        """
        Get the A and B matrices from the M step.

        Parameters
        ----------
        xr, yr : array, shape (n_p, n_t)
            X and Y positions of camera relative to image.
        w : array, shape (n_p, n_t)
            Weights associated with the positions.
        m : array, shape (n_sensors, n_t)
            Measurements.

        Returns
        -------
        a : array, shape (n_pix, n_pix)
            Inverse covariance of estimate
        b : array, shape (n_pix,)
            Bias term of loss.
        """
        feed_dict = {
            self.t_xr: xr,
            self.t_yr: yr,
            self.t_w: w,
            self.t_m: m
        }
        with self.sess.as_default():
            a, b = self.sess.run([self.t_a, self.t_b], feed_dict=feed_dict)
        return a, b

    def get_e(self, xr, yr, m, s):
        """
        Get the energy batched over data.

        Parameters
        ----------
        xr, yr : array, shape (n_p, n_t)
            X, Y positions of camera relative to image.
        m : array, shape (n_sensors, n_t)
            Measurements.
        s : array, shape (n_pix,)
            Current image estimate.

        Returns
        -------
        e : array, shape (n_p, n_t)
            log p(M|X, S
        """
        feed_dict = {
            self.t_xr: xr,
            self.t_yr: yr,
            self.t_m: m,
            self.t_s_inf: s
        }
        with self.sess.as_default():
            e = self.sess.run(self.t_e, feed_dict=feed_dict)
        return e


class GaussObsLP(LikelihoodPotential):
    """Likelihood potential for this application."""

    def __init__(self, D_H, D_O, energy_func=None):
        LikelihoodPotential.__init__(self, D_H, D_O)
        self.energy_func = energy_func

    def prob(self, Yc, Xc):
        """
        Get likelihood p(M|X, S)

        Parameters
        ----------
        Yc : array, shape (D_0,)
        Xc : array, shape (N_S, D_H)

        Returns
        -------
        Ps : array, shape (N_S,)
        """
        xr = Xc[:, 0:1]
        yr = Xc[:, 1:2]
        m = Yc[:, np.newaxis]

        E = self.energy_func(xr, yr, m)[:, 0]
        E = E - E.min()
        return np.exp(-E)
