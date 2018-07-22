"""
Main script.

(1) Generate spikes given an image and an eye path
(2) Use an EM-like algorithm to infer the eye path and image from spikes

"""

import os
import numpy as np

from utils.path_generator import (DiffusionPathGenerator,
                                  ExperimentalPathGenerator)
import utils.particle_filter as pf
from utils.time_filename import time_string
from utils.BurakPoissonLP import PoissonLP
from utils.hex_lattice import gen_hex_lattice
from src.theano_backend import TheanoBackend
from utils.h5py_utils import save_dict


class EMBurak(object):
    """Produce spikes and infers the causes that generated those spikes."""

    def __init__(
        self,
        l_i,
        d,
        motion_gen,
        motion_prior,
        dt=0.001,
        n_t=50,
        l_n=14,
        neuron_layout='sqr',
        drop_prob=None,
        l0=10.,
        l1=100.,
        ds=1.,
        de=1.,
        lamb=0.,
        tau=1.28,
        save_mode=False,
        n_itr=20,
        s_gen_name=' ',
        output_dir_base='',
        n_g_itr=40,
        fista_c=0.8,
        n_p=20,
        print_mode=True,
        gamma=10.,
        quad_reg=None,
        quad_reg_mean=None,
        s_range='pos',
        save_pix_rf_coupling=False,
        save_hessian=False,
    ):
        """
        Initialize the parts of the EM algorithm.

            -- Sets all parameters
            -- Compiles the theano backend
            -- Sets the gain factor for the spikes
            -- Initializes the object that generates the paths
            -- Initializes the Particle Filter object
            -- Checks that the output directory exists

        Parameters
        ----------
        l_i: int
            Length of pattern in pixels.
        d : array, float32, shape (n_l, n_pix)
            Dictionary used to infer latent factors
        s_gen_name : str
            Name of image (eg. label)
        dt : float
            Timestep for Simulation
        tau : float
            Decay constant for hessian
        ds : float
            Spacing between pixels of the image
        de : float
            Spacing between neurons
        n_t : int
            Number of timesteps of Simulation
        l_n : int
            Linear dimension of neuron array
        neuron_layout : str
            Either 'sqr' or 'hex' for a square or hexagonal grid
        drop_prob : float
            Probability of dropping out neurons from grid (None drops nothing)
        lamb: float
            Strength of sparse prior
        save_mode : bool
            True if you want to save the data
        n_itr : int
            Number of iterations to break the EM into
        s_gen_name : str
            Name for the generating image
        motion_gen : dict
            Dictionary containing:
            mode: str
                Either Diffusion or Experiment
            dc_gen : float
                Diffusion constant for generating the path
        motion_prior : dict
            Dictionary containing:
            'mode': str
                either PositionDiffusion or VelocityDiffusion
            dc : float
                Position Diffusion Constant
            dcv : float
                Velocity Diffusion Constant
            v0 : float
                Initial velocity for Velocity Diffusion Model
        output_dir_base : str
            Files saved to 'output/output_dir_base' If none, uses a time string
        quad_reg : array, shape (n_sp,)
            Prior on sparse coefficients A_i^2 * quad_reg[i]

        Note that changing certain parameters without reinitializing the class
        may have unexpected effects (because the changes won't necessarily
        propagate to subclasses.
        """
        self.data = {}
        self.save_mode = save_mode
        self.print_mode = print_mode
        self.save_pix_rf_coupling = save_pix_rf_coupling
        self.save_hessian = save_hessian

        if s_range == 'pos':
            smin, smax = 0, 1
            pos_only = True
        elif s_range == 'sym':
            smin, smax = -0.5, 0.5
            pos_only = False
        else:
            raise ValueError('Invalid s_range {}'.format(s_range))
        self.s_range = s_range

        d = d.astype('float32')

        # Assumes that the first dimension is 'Y'
        #    and the second dimension is 'X'
        self.s_gen_name = s_gen_name

        print 'The save mode is {}'.format(save_mode)

        # Simulation Parameters
        self.dt = dt  # Simulation timestep
        self.l0 = l0
        self.l1 = l1

        # Problem Dimensions
        self.n_t = n_t  # Number of time steps
        self.l_n = l_n  # Linear dimension of neuron receptive field grid
        self.n_l, n_pix = d.shape  # number of latent factors, image pixels
        self.l_i = l_i  # Linear dimension of the image

        if quad_reg is None:
            quad_reg = np.zeros((self.n_l,)).astype('float32')
        #  assert quad_reg.shape == (self.n_l,)
        self.quad_reg = quad_reg

        if quad_reg_mean is None:
            quad_reg_mean = np.zeros((self.n_l,)).astype('float32')
        assert quad_reg_mean.shape == (self.n_l,)
        self.quad_reg_mean = quad_reg_mean

        if not self.l_i ** 2 == n_pix:
            raise ValueError('Mismatch between dictionary and image size')

        self.ds = ds
        self.de = de

        # Image Prior Parameters
        self.gamma = gamma  # Pixel out of bounds cost parameter
        self.lamb = lamb  # the sparse prior is delta (S-DA) + lamb * |A|

        # EM Parameters
        # M - parameters (FISTA)
        self.fista_c = fista_c  # Constant to multiply fista L
        self.n_g_itr = n_g_itr
        self.n_itr = n_itr
        self.tau = tau  # Decay constant for summing hessian

        # E Parameters (Particle Filter)
        self.n_p = n_p  # Number of particles for the EM

        self.neuron_layout = neuron_layout
        (self.n_n, XE, YE, IE, XS, YS) = self.init_pix_rf_centers(
            l_n, self.l_i, ds, de, mode=neuron_layout, drop_prob=drop_prob)

        if drop_prob is None:
            drop_prob = 0.
        self.drop_prob = drop_prob

        # Variances of Gaussians for each pixel
        var = np.ones((self.l_i,)).astype('float32') * (
            (0.5 * ds) ** 2 + (0.203 * de) ** 2)

        self.tc = TheanoBackend(
            XS=XS,
            YS=YS,
            XE=XE,
            YE=YE,
            IE=IE,
            Var=var,
            d=d,
            l0=self.l0,
            l1=self.l1,
            DT=self.dt,
            G=1.,
            GAMMA=self.gamma,
            LAMBDA=self.lamb,
            TAU=self.tau,
            QUAD_REG=quad_reg,
            QUAD_REG_MEAN=quad_reg_mean,
            pos_only=pos_only,
            SMIN=smin,
            SMAX=smax
        )

        self.set_gain_factor((l_i, l_i))

        if motion_gen['mode'] == 'Diffusion':
            self.pg = DiffusionPathGenerator(
                self.n_t, self.l_i, motion_gen['dc'], self.dt)
        elif motion_gen['mode'] == 'Experiment':
            self.pg = ExperimentalPathGenerator(
                self.n_t, motion_gen['fpath'], self.dt)
        else:
            raise ValueError(
                'motion_gen[mode] must be Diffusion of Experiment')
        self.motion_gen = motion_gen

        self.pf = self.init_particle_filter(motion_prior, self.n_p)
        self.motion_prior = motion_prior

        if self.save_mode:
            self.output_dir = self.init_output_dir(output_dir_base)

        print 'Initialization done'

    def gen_data(self, s_gen, pg=None):
        """
        Generate a path and spikes.

        Builds a dictionary saving these data

        Parameters
        ----------
        s_gen : array, shape (l_i, l_i)
            Image that generates the spikes
        pg : PathGenerator
            Instance of path generator

        Returns
        -------
        xr : array, shape (1, n_t)
            X-Position of path generating spikes
        yr : array, shape (1, n_t)
            Y-Position of path generating spikes
        R : array, shape (n_n, n_t)
            Array containing the spike train for each neuron and timestep
        """
        # Generate Path
        if pg is None:
            pg = self.pg
        path = pg.gen_path()
        xr = path[0][np.newaxis, :].astype('float32')
        yr = path[1][np.newaxis, :].astype('float32')

        s_gen = s_gen.astype('float32')
        self.calculate_inner_products(s_gen, xr, yr)

        r = self.tc.spikes(s_gen, xr, yr)[0]
        if self.print_mode:
            print 'The mean firing rate is {:.2f}'.format(
                r.mean() / self.dt)

        if self.save_mode:
            self.build_param_and_data_dict(s_gen, xr, yr, r)

        return xr, yr, r

    @staticmethod
    def init_pix_rf_centers(l_n, l_i, ds, de, mode='sqr', drop_prob=None):
        """
        Initialize the centers of the receptive fields of the neurons.

        Parameters
        ----------
        l_n : int
            Length of the neuron array
        l_i : int
            Length of image array
        ds : float
            Spacing between pixels
        de : float
            Spacing between neurons
        mode : str
            Generate neuron array either a square grid or a hexagonal grid
        drop_prob : float
            Probability of dropping out a neuron (if None, drop nothing)

        Returns
        -------
        n_n : int
            Number of neurons
        xe : float array, shape (n_n,)
            X Coordinate of neuron centers
        ye : float array, (n_n,)
            Y Coordinate of neuron centers
        ie : float array, (n_n,)
            Identity of neurons (0 = ON, 1 = OFF)
        xs : float array, (l_i,)
            X Coordinate of Pixel centers
        ys : float array, (l_i,)
            Y Coordinate of Pixel centers
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

        if drop_prob is not None:
            idx = np.where(
                np.random.binomial(n=1, p=drop_prob, size=xe.size) == 0)
            xe, ye = xe[idx], ye[idx]

        xe = np.concatenate((xe, xe))
        ye = np.concatenate((ye, ye))
        n_n = xe.size

        # Identity of LGN cells (ON = 0, OFF = 1)
        ie = np.zeros((n_n,)).astype('float32')
        ie[0: n_n / 2] = 1

        # Position of pixels
        tmp = np.arange(l_i) - (l_i - 1) / 2.
        xs = ds * tmp.astype('float32')
        ys = ds * tmp.astype('float32')

        return n_n, xe, ye, ie, xs, ys

    def set_gain_factor(self, s_gen_shape):
        """
        Set the gain factor.

        An image with pixels of intensity 1
            results in spikes at the maximum firing rate.
        """
        g = 1.
        self.tc.set_gain_factor(g)

        Ips, _ = self.tc.RFS(
            np.ones(s_gen_shape).astype('float32'),
            np.zeros((1, self.n_t)).astype('float32'),
            np.zeros((1, self.n_t)).astype('float32'))
        g = (1. / Ips.max()).astype('float32')
        self.tc.set_gain_factor(g)

    def init_particle_filter(self, motion_prior, n_p):
        """
        Initialize the particle filter class.

        Parameters
        ----------
        motion_prior : dict
            Dictionary containing:
            'mode': str
                either PositionDiffusion or VelocityDiffusion
            dc : float
                Position Diffusion Constant
            dcv : float
                Velocity Diffusion Constant
            v0 : float
                Initial velocity for Velocity Diffusion Model
        n_p : int
            Number of particles for particle filter

        Returns
        -------
        pf : ParticleFilter
            Instance of particle filter
        """
        # Define necessary components for the particle filter
        if motion_prior['mode'] == 'PositionDiffusion':
            # Diffusion
            dc_infer = motion_prior['dc']
            d_h = 2  # Dimension of hidden state (i.e. x,y = 2 dims)
            sdev = np.sqrt(dc_infer * self.dt / 2) * np.ones((d_h,))
            ipd = pf.GaussIPD(d_h, self.n_n, sdev * 0.001)
            tpd = pf.GaussTPD(d_h, self.n_n, sdev)
            ip = pf.GaussIP(d_h, sdev * 0.001)
            tp = pf.GaussTP(d_h, sdev)
            lp = PoissonLP(self.n_n, d_h, self.tc.spike_energy)

        elif motion_prior['mode'] == 'VelocityDiffusion':
            # FIXME: save these params
            d_h = 4   # Hidden state dim, x,y,vx,vy

            v0 = motion_prior['v0']  # Initial Estimate for velocity
            dcv = motion_prior['dcv']  # Velocity Diffusion Constant
            st = np.sqrt(dcv * self.dt)
            adj = np.sqrt(1 - st ** 2 / v0 ** 2)

            eps = 0.00001  # Small number since cannot have exact zero
            sigma0 = np.array([eps, eps, v0, v0])  # Initial sigmas
            sigma_t = np.array([eps, eps, st, st])  # Transition sigmas

            # Transition matrix
            a = np.array([[1, 0, self.dt, 0],
                          [0, 1, 0, self.dt],
                          [0, 0, adj, 0],
                          [0, 0, 0, adj]])

            ipd = pf.GaussIPD(d_h, self.n_n, sigma0)
            tpd = pf.GaussTPD(d_h, self.n_n, sigma_t, A=a)
            ip = pf.GaussIP(d_h, sigma0)
            tp = pf.GaussTP(d_h, sigma_t, A=a)
            lp = PoissonLP(self.n_n, d_h, self.tc.spike_energy)
            # Note trick where PoissonLP takes 0,1 components of the
            # hidden state which is the same for both cases

        else:
            raise ValueError(
                'Unrecognized Motion Prior ' + str(motion_prior))

        r = np.zeros((self.n_n, self.n_t)).astype('float32')
        return pf.ParticleFilter(
            ipd, tpd, ip, tp, lp, r.transpose(), n_p)

    def reset(self):
        """Reset the class between EM runs."""
        self.data = {}
        self.pf.reset()

        self.tc.reset()
        # Reset the neuron grid
        (self.n_n, XE, YE, IE, _, _) = self.init_pix_rf_centers(
            self.l_n, self.l_i, self.ds, self.de, mode=self.neuron_layout)
        self.tc.t_XE.set_value(XE)
        self.tc.t_YE.set_value(YE)
        self.tc.t_IE.set_value(IE)
        self.pf = self.init_particle_filter(self.motion_prior, self.n_p)

    def run_e(self, t):
        """
        Run the the particle filter until it has run a total of t time steps.

        t - number of timesteps
        The result is saved in self.pf.XS,WS,means
        """
        if t > self.n_t:
            raise IndexError('Maximum simulated timesteps exceeded in E step')
        if self.pf.t >= t:
            raise IndexError(
                'Particle filter already run past given time point')
        while self.pf.t < t:
            self.pf.advance()
        self.pf.calculate_means_sdevs()

        # print 'Path SNR ' + str(SNR(self.xr[0][0:t], self.pf.means[0:t, 0]))

    def run_m(self, t0, tf, r, n_g_itr=5):
        """
        Run the maximization step.

        Parameters
        ----------
        t0, tf: int
            Starting and ending times of spike batch.
        r : float array, shape (n_n, n_t)
            All spikes for the given trial.
        """
        r_ = r[:, t0:tf]
        xr = self.pf.XS[t0:tf, :, 0].transpose()
        yr = self.pf.XS[t0:tf, :, 1].transpose()
        w = self.pf.WS[t0:tf].transpose()
        self._run_m(t0, tf, r_, xr, yr, w, n_g_itr=n_g_itr)

    def run_m_true_path(self, t0, tf, r, xr, yr, n_g_itr=5):
        """
        Run m step knowing true path.

        Parameters
        ----------
        t0, tf: int
            Starting and ending times of spike batch.
        r : float array, shape (n_n, n_t)
            All spikes for the given trial.
        xr, yr : float array, shape (1, n_t)
            X and Y coordinates of true eye path
        """
        xr = xr[:, t0:tf]
        yr = yr[:, t0:tf]
        w = np.ones_like(xr).astype('float32')
        r_ = r[:, t0:tf]
        self._run_m(t0, tf, r_, xr, yr, w, n_g_itr=n_g_itr)

    def _run_m(self, t0, tf, r, xr, yr, w, n_g_itr=5):
        """
        Run the maximization step for the given batch of spikes.

        Resets the values of auxillary gradient descent variables at start
        Result is saved in t_A.get_value()

        Parameters
        ----------
        r : float array, shape (n_n, tf - t0)
            Spikes
        xr, yr : float array, shape (n_b, tf - t0)

        w : float array, shape (n_b, tf - t0)

        n_g_itr : int
            Number of gradient steps
        """
        self.tc.init_m_aux()
        self.tc.update_Ap()

        if self.print_mode:
            desc = ''
            for item in ['E', 'E_prev', 'E_R', 'E_bnd', 'E_sp', 'E_lp',]:
                desc += '    {:<6} |'.format(item)
            desc += ' / Delta t'
            print desc

        fista_l = self.tc.calculate_L(
            tf, self.n_n, self.l0, self.l1, self.dt, self.fista_c)

        for v in range(n_g_itr):
            es = self.tc.run_fista_step(xr, yr, r, w, fista_l)
            #  self.img_SNR = 0.  # SNR(self.s_gen, self.tc.image_est())

            if self.print_mode:
                if v == 0:
                    es0 = es
                if v % 50 == 0:
                    des = [Ei - E0 for Ei, E0 in zip(es, es0)]
                    print self.get_cost_string(des, tf - t0)

        self.tc.update_HB(xr, yr, w)
        if self.print_mode:
            print 'The hessian trace is {}'.format(
                np.trace(self.tc.t_H.get_value()))

    @staticmethod
    def get_cost_string(es, t):
        """
        Print costs given output of img_grad.

        Cost divided by the number of timesteps
        es - tuple containing the differet costs
        t - number to timesteps
        """
        strg = ''
        for item in es:
            strg += '{:011.7f}'.format(item / t) + ' '
        return strg

    def run_em(self, r):
        """
        Run full expectation maximization algorithm.

        Saves summary of run info in self.data
        Note running twice will overwrite this run info

        Parameters
        ----------
        r : array, shape (n_n, n_t)
            Spike train to decode
        self.n_itr : int
            Number of iterations of EM
        self.n_g_itr : int
            Number of gradient steps in M step
        """
        self.tc.reset()
        self.pf.Y = r.T
        em_data = {}
        if self.print_mode:
            print 'The hessian trace is {}'.format(
                np.trace(self.tc.t_H.get_value()))

        print 'Running full EM'

        for u in range(self.n_itr):
            t0 = self.n_t * u / self.n_itr
            tf = self.n_t * (u + 1) / self.n_itr
            print('Iteration: {} | Running up to time {}'.format(u, tf))

            self.run_e(tf)
            self.run_m(t0, tf, r, n_g_itr=self.n_g_itr)

            iteration_data = {
                'time_steps': tf,
                'path_means': self.pf.means,
                'path_sdevs': self.pf.sdevs,
                'image_est': self.tc.image_est(),
                'coeff_est': self.tc.get_A()
            }

            if self.save_pix_rf_coupling:
                xr = self.pf.XS[t0:tf, :, 0].transpose()
                yr = self.pf.XS[t0:tf, :, 1].transpose()
                w = self.pf.WS[t0:tf].transpose()
                tmp = self.tc.get_sp_rf_coupling(xr, yr, w)
                iteration_data['pix_rf_coupling'] = tmp

            if self.save_hessian:
                iteration_data['hessian'] = self.tc.t_H.get_value()


            em_data[u] = iteration_data
        em_data['mode'] = 'EM'

        if self.save_mode:
            self.data['EM_data'] = em_data

    def run_inference_no_motion(self, r, rho=0.8, eps=0.01):
        """
        Run inference for pattern assuming no motion.

        Parameters
        ----------
        r : array, shape (n_n, n_t)
            Spikes

        """
        em_data = {}
        n_n, n_t = r.shape
        print('Running Inference assuming No Motion')
        print('E_no_mo')

        for u in range(self.n_itr):
            #  t0 = self.n_t * u / self.n_itr
            tf = self.n_t * (u + 1) / self.n_itr

            xr = yr = np.zeros((1, 1), dtype='float32')
            r_av = np.mean(r[:, 0:tf], axis=1, keepdims=True).astype('float32')

            print('Iteration: {} | Running up to time {}'.format(u, tf))
            self.tc.reset_adadelta_variables()
            for v in range(self.n_g_itr):
                es = self.tc.run_image_max_step(xr, yr, r_av, rho, eps)
                if self.print_mode:
                    if v == 0:
                        es0 = es
                    if v % 20 == 0:
                        des = [Ei - E0 for Ei, E0 in zip(es, es0)]
                        print self.get_cost_string(des, 1)

            iteration_data = {
                'time_steps': n_t,
                #  'path_means': self.pf.means,
                #  'path_sdevs': self.pf.sdevs,
                'image_est': self.tc.image_est(),
                'coeff_est': self.tc.get_A()
            }

            em_data[str(u)] = iteration_data
        em_data['mode'] = 'NoMotion'

        if self.save_mode:
            self.data['EM_data'] = em_data

    def run_inference_true_path(self, r, xr, yr):
        """
        Run inference algorithm given the true eye path.

        Parameters
        ----------
        r : array, shape (n_t, n_n)
            Spikes
        xr, yr : float array, shape (1, n_t)
            True X and Y position of the eye
        """
        self.tc.reset()

        em_data = {}

        print('Running Image Optimization using True Eye Path\n')

        for u in range(self.n_itr):
            t0 = self.n_t * u / self.n_itr
            tf = self.n_t * (u + 1) / self.n_itr
            print('Iteration: {} | Running up to time {}'.format(u, tf))

            self.run_m_true_path(t0, tf, r, xr, yr, n_g_itr=self.n_g_itr)

            iteration_data = {
                'time_steps': tf,
                'image_est': self.tc.image_est(),
                'coeff_est': self.tc.get_A()}

            em_data[u] = iteration_data
        em_data['mode'] = 'path_given'

        if self.save_mode:
            self.data['EM_data'] = em_data

    @staticmethod
    def init_output_dir(output_dir_base):
        """
        Create the output directory: output/output_dir_base.

        Parameters
        ----------
        output_dir_base : str

        Returns
        -------
        output_dir : str
            Output directory
        """
        if output_dir_base is None:
            output_dir_base = time_string()
        output_dir = os.path.join('output/', output_dir_base)
#        if not os.path.exists('output'):
#            os.mkdir('output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return output_dir

    def build_param_and_data_dict(self, s_gen, xr, yr, r):
        """
        Create a dictionary.

        self.data, that has all of the parameters of the model
        """
        # Note it is important to create a new dictionary here so that
        # we reset the data dict after generating new data
        self.data = {
            'DT': self.dt,
            'motion_prior': self.motion_prior,
            'motion_gen': self.motion_gen,
            'ds': self.ds,
            'de': self.de,
            'L0': self.l0,
            'L1': self.l1,
            'GAMMA': self.gamma,
            'lamb': self.lamb,
            'fista_c': self.fista_c,
            'D': self.tc.t_D.get_value(),
            'N_L': self.n_l,
            'N_T': self.n_t,
            'L_I': self.l_i,
            'L_N': self.l_n,
            'N_g_itr': self.n_g_itr,
            'N_itr': self.n_itr,
            'N_P': self.n_p,
            'XS': self.tc.t_XS.get_value(),
            'YS': self.tc.t_YS.get_value(),
            'XE': self.tc.t_XE.get_value(),
            'YE': self.tc.t_YE.get_value(),
            'Var': self.tc.t_Var.get_value(),
            'G': self.tc.t_G.get_value(),
            'tau': self.tau,
            'XR': xr, 'YR': yr,
            'IE': self.tc.t_IE.get_value(),
            'S_gen': s_gen,
            'S_gen_name': self.s_gen_name,
            'R': r,
            'Ips': self.Ips,
            'FP': self.FP,
            'quad_reg': self.quad_reg,
            'quad_reg_mean': self.quad_reg_mean,
            'drop_prob': self.drop_prob,
            's_range': self.s_range,
        }

    def save(self, compute_snrs=True):
        """
        Save information relevant to the EM run.

        data.pkl - saves dictionary with all data relevant to EM run
        (Only includes dict for EM data if that was run)
        Returns the filename
        """
        if not self.save_mode:
            raise RuntimeError('Need to enable save mode to save')

        fn = os.path.join(self.output_dir,
                          'data_' + time_string() + '.h5')
        save_dict(fn=fn, d=self.data)
        if compute_snrs:
            from src.analyzer import DataAnalyzer
            da = DataAnalyzer.fromfilename(fn)
            da.snr_list()
        return fn

    def calculate_inner_products(self, s_gen, xr, yr):
        """Calculate the inner products used."""
        self.Ips, self.FP = self.tc.RFS(s_gen, xr, yr)

    # Debug methods

    def get_hessian(self):
        """Return the hessian of the spike likelihood term."""
        return self.tc.hessian_func(
            self.pf.XS[:, :, 0].transpose(),
            self.pf.XS[:, :, 1].transpose(),
            self.pf.WS[:].transpose())

    def get_spike_cost(self, r):
        """Get -log p(R|X, S=DA)."""
        return self.tc.costs(
            self.pf.XS[:, :, 0].transpose(),
            self.pf.XS[:, :, 1].transpose(),
            r,
            self.pf.WS[:].transpose())[2]

    def get_sp_rf_test(self):
        u, v = self.tc.sp_rf_test(
            self.pf.XS[:, :, 0].transpose(),
            self.pf.XS[:, :, 1].transpose(),
            self.pf.WS[:].transpose())
        return u, v

    def ideal_observer_cost(self, xr, yr, r, s):
        """
        Get p(R|X, S).

        Parameters
        ----------
        r : array, shape (n_n, n_t)
            Spikes
        s : array, shape (l_i, l_i)
            Image that generates the spikes
        xr, yr: array, shape (1, n_t)
            Locations of eye that generated data
        Returns
        -------
        cost : float
            -log p(R|X, S)
        """
        w = np.ones_like(xr)
        return self.tc.image_costs(xr, yr, r, w, s)


#  from utils.gradient_checker import hessian_check

#  def f(A):
#     emb.tc.t_A.set_value(A.astype('float32'))
#     return emb.get_spike_cost()


#  def fpp(A):
#     emb.tc.t_A.set_value(A.astype('float32'))
#     return emb.get_hessian()

#  x0 = emb.tc.get_A()

#  for _ in range(2):
#     u, v = hessian_check(f, fpp, (D.shape[0],), x0=x0)
#     print u, v


# def true_path_infer_image_costs(self, N_g_itr = 10):
#        """
#       Infers the image given the true path
#        Prints the costs associated with this step
#        """
#        self.reset_img_gpu()
#        print 'Original Path, infer image'
#        t = self.N_T
#        self.run_m(t)


#    def true_image_infer_path_costs(self):
#        print 'Original image, Infer Path'
#        print 'Path SNR'
#        self.t_S.set_value(self.S)
#        for _ in range(4):
#            self.run_e(self.N_T)
#
#        if self.debug:
#            self.pf.plot(self.xr[0], self.yr[0], self.DT)
