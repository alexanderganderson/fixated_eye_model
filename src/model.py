# Python script containing a class that does expectation maximization
#   to estimate an image from simulated LGN cell responses
# See the end of the script for a sample usage

import numpy as np
import os
import cPickle as pkl

from utils.path_generator import (DiffusionPathGenerator,
                                  ExperimentalPathGenerator)
import utils.particle_filter_new as pf
from utils.SNR import SNR
from utils.time_filename import time_string
from utils.BurakPoissonLP import PoissonLP
from src.theano_backend import TheanoBackend


class EMBurak(object):

    def __init__(self, s_gen, d, dt=0.001,
                 n_t=50,
                 l_n=14, a=1., lamb=0.,
                 save_mode=False,
                 n_itr=20, s_gen_name=' ',
                 motion_gen_mode='Diffusion', dc_gen=100.,
                 motion_prior='PositionDiffusion', dc_infer=100.,
                 output_dir=''):
        """
        Initializes the parts of the EM algorithm
            -- Sets all parameters
            -- Initializes Dictionary
            -- Compiles the theano backend
            -- Sets the gain factor for the spikes
            -- Initializes the object that generates the paths
            -- Initializes the Particle Filter object
            -- Checks that the output directory exists

        Parameters
        ----------
        s_gen : array, float32, shape (l_i, l_i)
            Image that generates the spikes -
        d : array, float32, shape (n_l, n_pix)
            Dictionary used to infer latent factors
        s_gen_name : str
            Name of image (eg. label)
        dt : float
            Timestep for Simulation
        n_t : int
            Number of timesteps of Simulation
        l_n : int
            Linear dimension of neuron array
        a : float
            Image pixel spacing / receptive field spacing
        lamb: float
            Strength of sparse prior
        save_mode : bool
            True if you want to save the data
        n_itr : int
            Number of iterations to break the EM into
        s_gen_name : str
            Name for the generating image
        motion_gen_mode : str
            Method to generate motion. Either Diffusion or Experiment
        dc_gen : float
            Diffusion constant for generating motion
        motion_prior : str
            Prior to use for infering motion
        dc_infer : float
            Diffusion constant for inference
        output_dir : str
            Files saved to 'output/output_dir' If none, uses a time string

        Checks for consistency L_I ** 2 = N_pix

        Note that changing certain parameters without reinitializing the class
        may have unexpected effects (because the changes won't necessarily
        propagate to subclasses.
        """

        self.data = {}

        if output_dir is None:
            output_dir = time_string()
        self.output_dir = os.path.join('output/', output_dir)

        self.save_mode = save_mode  # If true, save results after EM completes
        print 'The save mode is ' + str(save_mode)
        self.d = d.astype('float32')  # Dictionary
        self.d_scl = np.sqrt((d ** 2).sum(1).mean())
        self.n_l, self.n_pix = d.shape
        # n_l - number of latent factors
        # n_pix - number of pixels in the image

        self.s_gen = s_gen.astype('float32')
        self.s_gen_name = s_gen_name
        self.l_i = s_gen.shape[0]  # Linear dimension of the image

        if not self.l_i ** 2 == self.n_pix:
            raise ValueError('Mismatch between dictionary and image size')

        # Simulation Parameters
        self.dt = dt  # Simulation timestep
        self.dc_gen = dc_gen  # Diffusion Constant for Generating motion
        self.dc_infer = dc_infer  # Diffusion Constant for Infering motion
        print 'The diffusion constant is {}'.format(self.dc_gen)
        self.l0 = 10.
        self.l1 = 100.

        # Problem Dimensions
        self.n_t = n_t  # Number of time steps
        self.l_n = l_n  # Linear dimension of neuron receptive field grid

        # Image Prior Parameters
        self.gamma = 100.  # Pixel out of bounds cost parameter
        self.lamb = lamb  # the sparse prior is delta (S-DA) + lamb * |A|

        # EM Parameters
        # M - Parameters (ADADELTA)
        # self.rho = 0.4
        # self.eps = 0.001

        # M - parameters (FISTA)
        self.fista_c = 0.8  # Constant to multiply fista L
        self.n_g_itr = 5
        self.n_itr = n_itr

        # E Parameters (Particle Filter)
        self.n_p = 5  # Number of particles for the EM

        # Initialize pixel and LGN positions
        self.a = a  # pixel spacing
        # Position of pixels
        self.XS = np.arange(- self.l_i / 2, self.l_i / 2)
        self.YS = np.arange(- self.l_i / 2, self.l_i / 2)
        self.XS = self.XS.astype('float32')
        self.YS = self.YS.astype('float32')
        self.XS *= self.a
        self.YS *= self.a

        # Position of LGN receptive fields
        self.init_rf_centers()

        # Variances of Gaussians for each pixel
        self.Var = 0.25 * np.ones((self.l_i,)).astype('float32')

        # Gain factor (to be set later)
        G = 1.

        # X-Position of retina (batches, timesteps), batches used for inference
        # only
        self.XR = np.zeros((1, self.n_t)).astype('float32')
        # Y-Position of retina
        self.YR = np.zeros((1, self.n_t)).astype('float32')

        # Spikes (1 or 0)
        self.R = np.zeros((self.n_n, self.n_t)).astype('float32')

        # Initial Value for Sparse Coefficients
        A0 = np.zeros((self.n_l,)).astype('float32')

        # Initial Value for Hessian
        H0 = np.zeros((self.n_l, self.n_l)).astype('float32')

        # Shapes of other variables used elsewhere

        # Pixel values for generating image (same shape as estimated image)
        #        self.s_gen = np.zeros((self.l_i, self.l_i)).astype('float32')
        # Assumes that the first dimension is 'Y'
        #    and the second dimension is 'X'

        # Weighting for batches and time points
        # self.Wbt = np.ones((self.n_b, self.n_t)).astype('float32')

        # Dictionary going from latent factors to image
        #       self.d = np.zeros((self.n_l, self.n_pix)).astype('float32')

        self.tc = TheanoBackend(
            self.XS, self.YS, self.XE, self.YE, self.IE, self.Var,
            A0, H0, self.d, self.l0, self.l1, self.dt, G,
            self.gamma, self.lamb)
        self.set_gain_factor()

        if motion_gen_mode == 'Diffusion':
            self.pg = DiffusionPathGenerator(
                self.n_t, self.l_i, self.dc_gen, self.dt)
        elif motion_gen_mode == 'Experiment':
            self.pg = ExperimentalPathGenerator(
                self.n_t, 'data/resampled_paths.mat', self.dt)
        else:
            raise ValueError('motion_gen_mode must'
                             'be Diffusion of Experiment')

        self.motion_prior = motion_prior
        self.init_particle_filter()

        if (self.save_mode):
            self.init_output_dir()

        print 'Initialization done'

    def gen_data(self):
        """
        Generates a path and spikes
        Builds a dictionary saving these data
        """
        # Generate Path
        path = self.pg.gen_path()
        self.XR[0, :] = path[0]
        self.YR[0, :] = path[1]

        self.calculate_inner_products()

        self.gen_spikes()
        self.pf.Y = self.R.transpose()  # Update reference to spikes for PF
        # TODO: EWW

        if self.save_mode:
            self.build_param_and_data_dict()

    def init_rf_centers(self):
        """
        Initialize the centers of the receptive fields of the neurons
        """
        self.n_n = 2 * self.l_n ** 2
        self.XE, self.YE = np.meshgrid(
            np.arange(- self.l_n / 2, self.l_n / 2),
            np.arange(- self.l_n / 2, self.l_n / 2)
        )

        self.XE = self.XE.ravel().astype('float32')
        self.YE = self.YE.ravel().astype('float32')
#        self.XE *= self.a
#        self.YE *= self.a

        def double_array(m):
            """
            m - 1d array to be doubled
            :rtype : array that is two copies of m concatenated
            """
            l = m.shape[0]
            res = np.zeros((2 * l,))
            res[0:l] = m
            res[l: 2 * l] = m
            return res

        self.XE = double_array(self.XE)
        self.YE = double_array(self.YE)

        # Identity of LGN cells (ON = 0, OFF = 1)
        self.IE = np.zeros((self.n_n,)).astype('float32')
        self.IE[0: self.n_n / 2] = 1

    def set_gain_factor(self):
        """
        Sets the gain factor so that an image with pixels of intensity 1
            results in spikes at the maximum firing rate
        """
        G = 1.
        self.tc.set_gain_factor(G)
        Ips, FP = self.tc.RFS(np.ones_like(self.s_gen), self.XR, self.YR)
        G = (1. / Ips.max()).astype('float32')
        self.tc.set_gain_factor(G)

    def init_particle_filter(self):
        """
        Initializes the particle filter class
        Requires spikes to already be generated
        """
        # Define necessary components for the particle filter
        if self.motion_prior == 'PositionDiffusion':
            # Diffusion
            D_H = 2  # Dimension of hidden state (i.e. x,y = 2 dims)
            sdev = np.sqrt(self.dc_infer * self.dt / 2) * np.ones((D_H,))
            ipd = pf.GaussIPD(D_H, self.n_n, sdev * 0.001)
            tpd = pf.GaussTPD(D_H, self.n_n, sdev)
            ip = pf.GaussIP(D_H, sdev * 0.001)
            tp = pf.GaussTP(D_H, sdev)
            lp = PoissonLP(self.n_n, D_H, self.tc.spike_energy)

        elif self.motion_prior == 'VelocityDiffusion':
            # FIXME: save these params
            D_H = 4   # Hidden state dim, x,y,vx,vy
            v0 = 30.  # Initial Estimate for velocity
            dcv = 6.  # Velocity Diffusion Constant
            st = np.sqrt(dcv * self.dt)

            eps = 0.00001  # Small number since cannot have exact zero
            sigma0 = np.array([eps, eps, v0, v0])  # Initial sigmas
            sigma_t = np.array([eps, eps, st, st])  # Transition sigmas

            # Transition matrix
            A = np.array([[1, 0, self.dt, 0],
                          [0, 1, 0, self.dt],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])

            ipd = pf.GaussIPD(D_H, self.n_n, sigma0)
            tpd = pf.GaussTPD(D_H, self.n_n, sigma_t, A=A)
            ip = pf.GaussIP(D_H, sigma0)
            tp = pf.GaussTP(D_H, sigma_t, A=A)
            lp = PoissonLP(self.n_n, D_H, self.l0, self.l1,
                           self.dt, self.G, self.spike_energy)

        else:
            raise ValueError(
                'Unrecognized Motion Prior ' + str(self.motion_prior))

        self.pf = pf.ParticleFilter(ipd, tpd, ip, tp, lp,
                                    self.R.transpose(), self.n_p)

    def gen_spikes(self):
        """
        Generate LGN responses given the path and the image
        """
        self.R[:, :] = self.tc.spikes(self.s_gen, self.XR, self.YR)[0]
        print 'Mean firing rate ' + str(self.R.mean() / self.dt)

    def true_costs(self):
        """
        Prints out the negative log-likelihood of the observed spikes given the
            image and path that generated them
        Note that the energy is normalized by the number of timesteps
        """
        print 'Pre-EM testing'
        out = self.tc.costs(self.XR, self.YR, self.R, self.Wbt)
        E, E_bound, E_R, E_rec, E_sp = out
        print ('Costs of underlying data ' +
               str((E / self.n_t, E_rec / self.n_t)))

    def reset(self):
        """
        Resets the class between EM runs
        """
        self.pf.reset()
        # self.c.reset()
        self.data = {}
        self.tc.reset_image_estimate()
        self.tc.reset_m_aux()

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

        print 'Path SNR ' + str(SNR(self.XR[0][0:t], self.pf.means[0:t, 0]))

    def run_M(self, t, N_g_itr=5):
        """
        Runs the maximization step for the first t time steps
        resets the values of auxillary gradient descent variables at start
        t - number of time steps
        result is saved in t_A.get_value()
        """
        self.tc.reset_m_aux()
        print ('Total Energy / t | Bound. Energy / t ' +
               '| Spike Energy / t | + Sparse E / t |  SNR')
        fista_l = self.tc.calculate_L(
            t, self.n_n, self.l0, self.l1, self.dt, self.d_scl, self.fista_c)

        for v in range(N_g_itr):
            out = self.tc.img_grad(
                self.pf.XS[0:t, :, 0].transpose(),
                self.pf.XS[0:t, :, 1].transpose(),
                self.R[:, 0:t], self.pf.WS[0:t].transpose(),
                fista_l)
            self.img_SNR = SNR(self.s_gen, self.tc.image_est())

            print self.print_costs(out, t) + str(self.img_SNR)

    @staticmethod
    def print_costs(out, t):
        """
        Prints costs given output of img_grad
        Cost divided by the number of timesteps
        out - tuple containing the differet costs
        t - number to timesteps
        """
        strg = ''
        for item in out:
            strg += str(item / t) + ' '
        return strg

    def run_EM(self):
        """
        Runs full expectation maximization algorithm
        self.N_itr - number of iterations of EM
        self.N_g_itr - number of gradient steps in M step
        Saves summary of run info in self.data
        Note running twice will overwrite this run info
        """
        self.tc.reset_image_estimate()

        EM_data = {}

        print '\n' + 'Running full EM'

        for u in range(self.n_itr):
            t = self.n_t * (u + 1) / self.n_itr
            print ('\n' + 'Iteration number ' + str(u) +
                   ' Running up time = ' + str(t))

            # Run E step
            self.run_E(t)

            # Run M step
            if u <= 2:
                c = 4
            else:
                c = 1
            self.run_M(t, N_g_itr=self.n_g_itr * c)

            iteration_data = {'time_steps': t, 'path_means': self.pf.means,
                              'path_sdevs': self.pf.sdevs,
                              'image_est': self.tc.image_est(),
                              'coeff_est': self.tc.get_A()}

            EM_data[u] = iteration_data

        if self.save_mode:
            self.data['EM_data'] = EM_data

    def init_output_dir(self):
        """
        Create an output directory
        """
        if not os.path.exists('output'):
            os.mkdir('output')

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def build_param_and_data_dict(self):
        """
        Creates a dictionary, self.data, that has all of the parameters
            of the model
        """
        # Note it is important to create a new dictionary here so that
        # we reset the data dict after generating new data
        self.data = {'DT': self.dt,
                     'DC_gen': self.dc_gen,
                     'DC_infer': self.dc_infer,
                     'L0': self.l0,
                     'L1': self.l1,
                     'GAMMA': self.gamma,
                     'lamb': self.lamb,
                     'D': self.d,
                     'N_L': self.n_l,
                     'a': self.a,
                     'N_T': self.n_t,
                     'L_I': self.l_i,
                     'L_N': self.l_n,
                     'N_g_itr': self.n_g_itr,
                     'N_itr': self.n_itr,
                     'N_P': self.n_p,
                     'XS': self.XS, 'YS': self.YS,
                     'XE': self.XE, 'YE': self.YE,
                     'Var': self.Var,
                     'G': self.tc.t_G.get_value(),
                     'XR': self.XR, 'YR': self.YR,
                     'IE': self.IE,
                     'actual_motion_mode': self.pg.mode(),
                     'S_gen': self.s_gen, 'S_gen_name': self.s_gen_name,
                     'R': self.R,
                     'Ips': self.Ips, 'FP': self.FP,
                     'motion_prior': self.motion_prior}

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

    def calculate_inner_products(self):
        """
        Calculates the inner products used
        """
        self.Ips, self.FP = self.tc.RFS(self.s_gen, self.XR, self.YR)

    def get_hessian(self):
        return self.tc.hessian_func(
            self.pf.XS[:, :, 0].transpose(),
            self.pf.XS[:, :, 1].transpose(),
            self.pf.WS[:].transpose())

    def get_costs(self):
        return self.tc.costs(
            self.pf.XS[:, :, 0].transpose(),
            self.pf.XS[:, :, 1].transpose(),
            self.R,
            self.pf.WS[:].transpose())

    def get_spike_cost(self):
        return self.tc.costs(
            self.pf.XS[:, :, 0].transpose(),
            self.pf.XS[:, :, 1].transpose(),
            self.R,
            self.pf.WS[:].transpose())[2]




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
