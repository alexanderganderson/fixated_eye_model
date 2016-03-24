"""
Main script.

(1) Generate spikes given an image and an eye path
(2) Use an EM-like algorithm to infer the eye path and image from spikes

"""

import numpy as np
import os
import cPickle as pkl

from utils.path_generator import (DiffusionPathGenerator,
                                  ExperimentalPathGenerator)
import utils.particle_filter as pf
from utils.time_filename import time_string
from utils.BurakPoissonLP import PoissonLP
from utils.hex_lattice import gen_hex_lattice
from src.theano_backend import TheanoBackend


class EMBurak(object):
    """Produce spikes and infers the causes that generated those spikes."""

    def __init__(
        self, s_gen, d, motion_gen, motion_prior,
        dt=0.001, n_t=50, l_n=14, neuron_layout='sqr',
        ds=1., de=1., lamb=0.,
        tau=0.05, save_mode=False, n_itr=20, s_gen_name=' ',
        output_dir_base=''
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
        s_gen : array, float32, shape (l_i, l_i), entries in [0, 1]
            Image that generates the spikes
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

        Note that changing certain parameters without reinitializing the class
        may have unexpected effects (because the changes won't necessarily
        propagate to subclasses.
        """
        self.data = {}
        self.save_mode = save_mode

        d = d.astype('float32')

        s_gen = s_gen.astype('float32')
        # Assumes that the first dimension is 'Y'
        #    and the second dimension is 'X'
        self.s_gen_name = s_gen_name

        print 'The save mode is {}'.format(save_mode)

        # Simulation Parameters
        self.dt = dt  # Simulation timestep
        self.l0 = 10.
        self.l1 = 100.

        # Problem Dimensions
        self.n_t = n_t  # Number of time steps
        self.l_n = l_n  # Linear dimension of neuron receptive field grid
        self.n_l, n_pix = d.shape  # number of latent factors, image pixels
        self.l_i = s_gen.shape[0]  # Linear dimension of the image

        if not self.l_i ** 2 == n_pix:
            raise ValueError('Mismatch between dictionary and image size')

        self.ds = ds
        self.de = de

        # Image Prior Parameters
        self.gamma = 100.  # Pixel out of bounds cost parameter
        self.lamb = lamb  # the sparse prior is delta (S-DA) + lamb * |A|

        # EM Parameters
        # M - parameters (FISTA)
        self.fista_c = 0.8  # Constant to multiply fista L
        self.n_g_itr = 5
        self.n_itr = n_itr
        self.tau = tau  # Decay constant for summing hessian

        # E Parameters (Particle Filter)
        self.n_p = 20  # Number of particles for the EM

        (self.n_n, XE, YE, IE, XS, YS, neuron_mode
         ) = self.init_pix_rf_centers(l_n, self.l_i, ds, de,
                                      mode=neuron_layout)

        # Variances of Gaussians for each pixel
        var = np.ones((self.l_i,)).astype('float32') * (
            (0.5 * ds) ** 2 + (0.203 * de) ** 2)

        self.tc = TheanoBackend(
            XS, YS, XE, YE, IE, var,
            d, self.l0, self.l1, self.dt, 1.,
            self.gamma, self.lamb, self.tau)
        self.set_gain_factor(s_gen.shape)

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

    def gen_data(self, s_gen, pg=None, print_mode=True):
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
        XR : array, shape (1, n_t)
            X-Position of path generating spikes
        YR : array, shape (1, n_t)
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

        self.calculate_inner_products(s_gen, xr, yr)

        R = self.tc.spikes(s_gen, XR, YR)[0]
        if print_mode:
            print 'The mean firing rate is {:.2f}'.format(
                R.mean() / self.dt)

        self.pf.Y = R.transpose()  # Update reference to spikes for PF
        # TODO: EWW

        if self.save_mode:
            self.build_param_and_data_dict(s_gen, XR, YR, R)

        return XR, YR, R

    @staticmethod
    def init_pix_rf_centers(l_n, l_i, ds, de, mode='sqr'):
        """
        Initialize the centers of the receptive fields of the neurons
            Creates a population of on cells and off cells

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

        Returns
        -------
        n_n : int
            Number of neurons
        XE : float array, shape (n_n,)
            X Coordinate of neuron centers
        YE : float array, (n_n,)
            Y Coordinate of neuron centers
        IE : float array, (n_n,)
            Identity of neurons (0 = ON, 1 = OFF)
        XS : float array, (l_i,)
            X Coordinate of Pixel centers
        YS : float array, (l_i,)
            Y Coordinate of Pixel centers
        """

        if mode == 'sqr':
            XE, YE = np.meshgrid(
                de * np.arange(- l_n / 2, l_n / 2),
                de * np.arange(- l_n / 2, l_n / 2))
            XE, YE = XE.ravel(), YE.ravel()
        elif mode == 'hex':
            XE, YE = gen_hex_lattice(l_n * de, a=de)
        else:
            raise ValueError('Unrecognized Neuron Mode {}'.format(mode))
        XE, YE = XE.astype('float32'), YE.astype('float32')

        XE = np.concatenate((XE, XE))
        YE = np.concatenate((YE, YE))
        n_n = XE.size

        # Identity of LGN cells (ON = 0, OFF = 1)
        IE = np.zeros((n_n,)).astype('float32')
        IE[0: n_n / 2] = 1

        # Position of pixels
        tmp = np.arange(l_i) - (l_i - 1) / 2.
        XS = ds * tmp.astype('float32')
        YS = ds * tmp.astype('float32')

        return n_n, XE, YE, IE, XS, YS, mode

    def set_gain_factor(self, s_gen_shape):
        """
        Sets the gain factor so that an image with pixels of intensity 1
            results in spikes at the maximum firing rate
        """
        G = 1.
        self.tc.set_gain_factor(G)

        Ips, FP = self.tc.RFS(
            np.ones(s_gen_shape).astype('float32'),
            np.zeros((1, self.n_t)).astype('float32'),
            np.zeros((1, self.n_t)).astype('float32'))
        G = (1. / Ips.max()).astype('float32')
        self.tc.set_gain_factor(G)

    def init_particle_filter(self, motion_prior, n_p):
        """
        Initializes the particle filter class

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
            D_H = 2  # Dimension of hidden state (i.e. x,y = 2 dims)
            sdev = np.sqrt(dc_infer * self.dt / 2) * np.ones((D_H,))
            ipd = pf.GaussIPD(D_H, self.n_n, sdev * 0.001)
            tpd = pf.GaussTPD(D_H, self.n_n, sdev)
            ip = pf.GaussIP(D_H, sdev * 0.001)
            tp = pf.GaussTP(D_H, sdev)
            lp = PoissonLP(self.n_n, D_H, self.tc.spike_energy)

        elif motion_prior['mode'] == 'VelocityDiffusion':
            # FIXME: save these params
            D_H = 4   # Hidden state dim, x,y,vx,vy

            v0 = motion_prior['v0']  # Initial Estimate for velocity
            dcv = motion_prior['dcv']  # Velocity Diffusion Constant
            st = np.sqrt(dcv * self.dt)
            adj = np.sqrt(1 - st ** 2 / v0 ** 2)

            eps = 0.00001  # Small number since cannot have exact zero
            sigma0 = np.array([eps, eps, v0, v0])  # Initial sigmas
            sigma_t = np.array([eps, eps, st, st])  # Transition sigmas

            # Transition matrix
            A = np.array([[1, 0, self.dt, 0],
                          [0, 1, 0, self.dt],
                          [0, 0, adj, 0],
                          [0, 0, 0, adj]])

            ipd = pf.GaussIPD(D_H, self.n_n, sigma0)
            tpd = pf.GaussTPD(D_H, self.n_n, sigma_t, A=A)
            ip = pf.GaussIP(D_H, sigma0)
            tp = pf.GaussTP(D_H, sigma_t, A=A)
            lp = PoissonLP(self.n_n, D_H, self.tc.spike_energy)
            # Note trick where PoissonLP takes 0,1 components of the
            # hidden state which is the same for both cases

        else:
            raise ValueError(
                'Unrecognized Motion Prior ' + str(motion_prior))

        R = np.zeros((self.n_n, self.n_t)).astype('float32')
        return pf.ParticleFilter(
            ipd, tpd, ip, tp, lp, R.transpose(), n_p)

    def reset(self):
        """
        Resets the class between EM runs
        """
        self.data = {}
        self.pf.reset()
        self.tc.reset()

    def run_E(self, t):
        """
        Runs the the particle filter until it has run a total of t time steps
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

        # print 'Path SNR ' + str(SNR(self.XR[0][0:t], self.pf.means[0:t, 0]))

    def run_M(self, t0, tf, R, N_g_itr=5):
        """
        Runs the maximization step for the first t time steps
        resets the values of auxillary gradient descent variables at start
        t - number of time steps
        result is saved in t_A.get_value()
        """
        self.tc.init_m_aux()
        self.tc.update_Ap()

        desc = ''
        for item in ['E', 'E_prev', 'E_R', 'E_bnd', 'E_sp', 'E_lp']:
            desc += '    {:<6} |'.format(item)
        desc += ' / Delta t | SNR'
        print desc

        fista_l = self.tc.calculate_L(
            tf, self.n_n, self.l0, self.l1, self.dt, self.fista_c)

        XR = self.pf.XS[t0:tf, :, 0].transpose()
        YR = self.pf.XS[t0:tf, :, 1].transpose()
        W = self.pf.WS[t0:tf].transpose()
        R_ = R[:, t0:tf]
        for v in range(N_g_itr):
            Es = self.tc.run_fista_step(XR, YR, R_, W, fista_l)
            self.img_SNR = 0.  # SNR(self.s_gen, self.tc.image_est())
            if v == 0:
                Es0 = Es
            dEs = [Ei - E0 for Ei, E0 in zip(Es, Es0)]
            print self.get_cost_string(dEs, tf - t0) + str(self.img_SNR)

        self.tc.update_HB(XR, YR, W)
        print 'The hessian trace is {}'.format(
            np.trace(self.tc.t_H.get_value()))

    @staticmethod
    def get_cost_string(Es, t):
        """
        Prints costs given output of img_grad
        Cost divided by the number of timesteps
        Es - tuple containing the differet costs
        t - number to timesteps
        """
        strg = ''
        for item in Es:
            strg += '{:011.7f}'.format(item / t) + ' '
        return strg

    def run_EM(self, R):
        """
        Runs full expectation maximization algorithm
        self.N_itr - number of iterations of EM
        self.N_g_itr - number of gradient steps in M step
        Saves summary of run info in self.data
        Note running twice will overwrite this run info

        Parameters
        ----------
        R : array, shape (n_n, n_t)
            Spike train to decode
        """
        self.tc.reset()

        EM_data = {}

        print 'Running full EM'

        for u in range(self.n_itr):
            t0 = self.n_t * u / self.n_itr
            tf = self.n_t * (u + 1) / self.n_itr
            print (
                '\nIteration number {} Running up to time {}'.format(u, tf))

            self.run_E(tf)

            c = 4  # if u <= 2 else 2
            self.run_M(t0, tf, R, N_g_itr=self.n_g_itr * c)

            iteration_data = {
                'time_steps': tf, 'path_means': self.pf.means,
                'path_sdevs': self.pf.sdevs,
                'image_est': self.tc.image_est(),
                'coeff_est': self.tc.get_A()}

            EM_data[u] = iteration_data

        if self.save_mode:
            self.data['EM_data'] = EM_data

    def init_output_dir(self, output_dir_base):
        """
        Create the output directory: output/output_dir_base

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
        if not os.path.exists('output'):
            os.mkdir('output')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        return output_dir

    def build_param_and_data_dict(self, s_gen, XR, YR, R):
        """
        Creates a dictionary, self.data, that has all of the parameters
            of the model
        """
        # Note it is important to create a new dictionary here so that
        # we reset the data dict after generating new data
        self.data = {'DT': self.dt,
                     'motion_prior': self.motion_prior,
                     'motion_gen': self.motion_gen,
                     'ds': self.ds,
                     'de': self.de,
                     'L0': self.l0,
                     'L1': self.l1,
                     'GAMMA': self.gamma,
                     'lamb': self.lamb,
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
                     'XR': XR, 'YR': YR,
                     'IE': self.tc.t_IE.get_value(),
                     'actual_motion_mode': self.pg.mode(),
                     'S_gen': s_gen, 'S_gen_name': self.s_gen_name,
                     'R': R,
                     'Ips': self.Ips, 'FP': self.FP}

    def save(self):
        """
        Saves information relevant to the EM run
        data.pkl - saves dictionary with all data relevant to EM run
        (Only includes dict for EM data if that was run)
        Returns the filename
        """
        if not self.save_mode:
            raise RuntimeError('Need to enable save mode to save')

        fn = os.path.join(self.output_dir,
                          'data_' + time_string() + '.pkl')
        pkl.dump(self.data, open(fn, 'wb'))
        return fn

    def calculate_inner_products(self, s_gen, XR, YR):
        """
        Calculates the inner products used
        """
        self.Ips, self.FP = self.tc.RFS(s_gen, XR, YR)

    """ Debug methods """

    def get_hessian(self):
        return self.tc.hessian_func(
            self.pf.XS[:, :, 0].transpose(),
            self.pf.XS[:, :, 1].transpose(),
            self.pf.WS[:].transpose())

    def get_spike_cost(self, R):
        return self.tc.costs(
            self.pf.XS[:, :, 0].transpose(),
            self.pf.XS[:, :, 1].transpose(),
            R,
            self.pf.WS[:].transpose())[2]

    def ideal_observer_cost(self, XR, YR, R, S):
        """
        Get p(R|X, S)

        Parameters
        ----------
        R : array, shape (n_n, n_t)
            Spikes
        S : array, shape (l_i, l_i)
            Image that generates the spikes
        XR, YR: array, shape (1, n_t)
            Locations of eye that generated data
        Returns
        -------
        cost : float
            -log p(R|X, S)
        """
        W = np.ones_like(XR)
        return self.tc.image_costs(XR, YR, R, W, S)


# from utils.gradient_checker import hessian_check

# def f(A):
#     emb.tc.t_A.set_value(A.astype('float32'))
#     return emb.get_spike_cost()


# def fpp(A):
#     emb.tc.t_A.set_value(A.astype('float32'))
#     return emb.get_hessian()

# x0 = emb.tc.get_A()

# for _ in range(2):
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
#        self.run_M(t)


#    def true_image_infer_path_costs(self):
#        print 'Original image, Infer Path'
#        print 'Path SNR'
#        self.t_S.set_value(self.S)
#        for _ in range(4):
#            self.run_E(self.N_T)
#
#        if self.debug:
#            self.pf.plot(self.XR[0], self.YR[0], self.DT)
