# theano_backend.py

import numpy as np
import theano
import theano.tensor as T
from utils.fista import fista_updates


class TheanoBackend(object):

    """
    Theano backend for executing the computations
    """

    def __init__(self, XS, YS, XE, YE, IE, Var, A0, H0, d,
                 l0, l1, DT, G, GAMMA, LAMBDA, pos_only=True):
        """
        Initializes all theano variables and functions
        """
        self.l_i = XS.shape[0]
        self.n_l = A0.shape[0]

        # Define Theano Variables Common to Generation and Inference
        self.t_XS = theano.shared(XS, 'XS')
        self.t_YS = theano.shared(YS, 'YS')
        self.t_XE = theano.shared(XE, 'XE')
        self.t_YE = theano.shared(YE, 'YE')
        self.t_IE = theano.shared(IE, 'IE')
        self.t_Var = theano.shared(Var, 'Var')

        self.t_XR = T.matrix('XR')
        self.t_YR = T.matrix('YR')
        self.t_R = T.matrix('R')

        #  Parameters
        self.t_L0 = theano.shared(np.float32(l0), 'L0')
        self.t_L1 = theano.shared(np.float32(l1), 'L1')
        self.t_DT = theano.shared(np.float32(DT), 'DT')
        self.t_G = theano.shared(np.float32(G), 'G')

        def inner_products(t_S, t_Var, t_XS, t_YS, t_XE, t_YE, t_XR, t_YR):
            """
            Take dot product of image with an array of gaussians
            t_S - image variable shape - (i2, i1)
            t_Var - variances of receptive fields
            t_XS - X coordinate for image pixels for dimension i1
            t_YS - Y coordinate for image pixels for dimension i2
            t_XE - X coordinate for receptive fields j
            t_YE - Y coordinate for receptive fields j
            t_XR - X coordinate for retina in form batch, timestep, b,t
            t_YR - Y coordinate ''

            Returns
            t_Ips - Inner products btw image and filters: b, j, t
            t_PixRFCoupling - d Ips / dS: b, i2, i1, j, t
            """

            # Note in this computation, we do the indices in this form:
            #  b, i, j, t
            #  batch, pixel, neuron, time step

            # indices: b, i1, j, t
            t_dX = (t_XS.dimshuffle('x', 0, 'x', 'x') -
                    t_XE.dimshuffle('x', 'x', 0, 'x') -
                    t_XR.dimshuffle(0, 'x', 'x', 1))
            t_dX.name = 'dX'
            # indices: b, i2, j, t
            t_dY = (t_YS.dimshuffle('x', 0, 'x', 'x') -
                    t_YE.dimshuffle('x', 'x', 0, 'x') -
                    t_YR.dimshuffle(0, 'x', 'x', 1))
            t_dY.name = 'dY'

            # Use outer product trick to dot product image with point filters
            t_PixRFCouplingX = T.exp(-0.5 * t_dX ** 2 /
                                     t_Var.dimshuffle('x', 0, 'x', 'x'))
            t_PixRFCouplingY = T.exp(-0.5 * t_dY ** 2 /
                                     t_Var.dimshuffle('x', 0, 'x', 'x'))
            t_PixRFCouplingX.name = 'PixRFCouplingX'
            t_PixRFCouplingY.name = 'PixRFCouplingY'

            # Matrix of inner products between the images and the retinal RFs
            # indices: b, j, t
            # Sum_i2 T(i2, i1) * T(b, i2, j, t) = T(b, i1, j, t)
            t_IpsY = T.sum(t_S.dimshuffle('x', 0, 1, 'x', 'x') *
                           t_PixRFCouplingY.dimshuffle(0, 1, 'x', 2, 3),
                           axis=1)
            # Sum_i1 T(b, i1, j, t) * T(b, i2, j, t) = T(b, j, t)
            t_Ips = T.sum(t_IpsY * t_PixRFCouplingX, axis=1)
            t_Ips.name = 'Ips'

            # For the gradient, we also prepare d Ips / dS
            # This is in the form b, i2, i1, j, t
            t_PixRFCoupling = (t_PixRFCouplingX.dimshuffle(0, 'x', 1, 2, 3) *
                               t_PixRFCouplingY.dimshuffle(0, 1, 'x', 2, 3))

            return t_Ips, t_PixRFCoupling

        def firing_prob(t_Ips, t_G, t_IE, t_L0, t_L1, t_DT):
            # Firing probabilities indexed by b, j, t
            # t_Ips - Image-RF inner products indexed as b, j, t
            # t_G - gain constant
            # t_IE - identity of retinal ganglion cells
            # t_L0, t_L1 - min, max firing rate
            # t_DT - time step size

            t_IEr = t_IE.dimshuffle('x', 0, 'x')
            t_Gen = t_IEr + (1 - 2 * t_IEr) * t_G * t_Ips  # Generator signal

            t_FP_0 = t_DT * T.exp(T.log(t_L0) + T.log(t_L1 / t_L0) * t_Gen)

            t_FP = t_FP_0
            return t_FP

        def dlogfp_dA(t_dIpsdA, t_G, t_IE, t_L0, t_L1):
            """
            t_dIpsdA - d Ips / dA indexed as b, k, j, t
            t_G - gain constant
            t_IE - RGC identity, j
            t_L0, t_L1 - min, max firing rates
            Returns the d log FP / dA indexed b, k, j, t

            """
            t_IEr = t_IE.dimshuffle('x', 'x', 0, 'x')
            # t_dGen_dA = t_IEr + (1 - 2 * t_IEr) * t_G * t_dIpsdA
            t_dGen_dA = (1 - 2 * t_IEr) * t_G * t_dIpsdA
            t_dlogFPdA = T.log(t_L1 / t_L0) * t_dGen_dA

            return t_dlogFPdA

        ##############################
        # Simulated Spike Generation #
        ##############################

        self.t_S_gen = T.matrix('S_gen')  # Image dims are i2, i1
        self.t_Ips_gen, _ = inner_products(self.t_S_gen, self.t_Var,
                                           self.t_XS, self.t_YS,
                                           self.t_XE, self.t_YE,
                                           self.t_XR, self.t_YR)
        self.t_FP_gen = firing_prob(self.t_Ips_gen, self.t_G, self.t_IE,
                                    self.t_L0, self.t_L1, self.t_DT)

        # Computes image-RF inner products and the resulting firing
        # probabilities
        self.RFS = theano.function(
            inputs=[self.t_S_gen, self.t_XR, self.t_YR],
            outputs=[self.t_Ips_gen, self.t_FP_gen])

        self.rng = T.shared_randomstreams.RandomStreams(seed=10)
        self.t_R_gen = (self.rng.uniform(size=self.t_FP_gen.shape) <
                        self.t_FP_gen).astype('float32')

        self.spikes = theano.function(
            inputs=[self.t_S_gen, self.t_XR, self.t_YR],
            outputs=self.t_R_gen)

        ##############################
        # Latent Variable Estimation #
        ##############################

        def spiking_cost(t_R, t_FP):
            """
            Returns the negative log likelihood of the spikes given
            the inner products
            t_R - spikes in form j, t
            t_FP - Firing probabilities in form b, j, t
            Returns -log p(R|X,S) with indices in the form b, j, t
            """
            #            t_E_R_f = -(t_R.dimshuffle('x', 0, 1) * T.log(t_FP)
            #         + (1 - t_R.dimshuffle('x', 0, 1)) * T.log(1 - t_FP))
            #         Try using poisson loss instead of bernoulli loss
            t_E_R_f = -t_R.dimshuffle('x', 0, 1) * T.log(t_FP) + t_FP

            t_E_R_f.name = 'E_R_f'

            return t_E_R_f

        self.spiking_cost = spiking_cost

        self.t_A = theano.shared(A0, 'A')
        self.t_D = theano.shared(d, 'D')
        self.t_S = T.dot(self.t_A, self.t_D).reshape((self.l_i, self.l_i))
        self.image_est = theano.function(inputs=[], outputs=self.t_S)
        # FIXME: shouldn't access L_I, Image dims are i2, i1

        self.t_GAMMA = theano.shared(np.float32(GAMMA), 'GAMMA')
        self.t_LAMBDA = theano.shared(np.float32(LAMBDA), 'LAMBDA')

        self.t_Ips, t_PixRFCoupling = inner_products(
            self.t_S, self.t_Var, self.t_XS, self.t_YS,
            self.t_XE, self.t_YE, self.t_XR, self.t_YR)

        self.t_FP = firing_prob(self.t_Ips, self.t_G, self.t_IE,
                                self.t_L0, self.t_L1, self.t_DT)

        # Compute Energy Functions (negative log-likelihood) to minimize
        # Weights (batch, timestep) from particle filter
        self.t_Wbt = T.matrix('Wbt')
        self.t_E_R_f = spiking_cost(self.t_R, self.t_FP)

        self.t_E_R = T.sum(T.sum(self.t_E_R_f, axis=1) * self.t_Wbt)
        self.t_E_R.name = 'E_R'

        self.t_E_bound = self.t_GAMMA * (
            T.sum(T.switch(self.t_S < 0., -self.t_S, 0)) +
            T.sum(T.switch(self.t_S > 1., self.t_S - 1, 0)))
        self.t_E_bound.name = 'E_bound'

        self.t_E_sp = self.t_LAMBDA * T.sum(T.abs_(self.t_A))
        self.t_E_sp.name = 'E_sp'

        self.t_E_rec = self.t_E_R + self.t_E_bound
        self.t_E_rec.name = 'E_rec'

        self.t_E = self.t_E_rec + self.t_E_sp
        self.t_E.name = 'E'

        # Cost from poisson terms separated by batches for particle filter log
        # probability
        self.t_E_R_b = T.sum(self.t_E_R_f, axis=(1, 2))
        self.spike_energy = theano.function(
            inputs=[self.t_XR, self.t_YR, self.t_R],
            outputs=self.t_E_R_b)

        # Generate costs given a path, spikes, and time-batch weights
        energy_outputs = [self.t_E, self.t_E_bound, self.t_E_R, self.t_E_sp]
        self.costs = theano.function(
            inputs=[self.t_XR, self.t_YR, self.t_R, self.t_Wbt],
            outputs=energy_outputs)

        # Define theano variables for gradient descent
        # self.t_Rho = T.scalar('Rho')
        # self.t_Eps = T.scalar('Eps')
        # self.t_ada_params = (self.t_Rho, self.t_Eps)

        # self.grad_updates = ada_delta(self.t_E, self.t_A, *self.t_ada_params)
        # self.t_A_Eg2, self.t_A_EdS2, _ = self.grad_updates.keys()

        # Define variables for FISTA minimization
        self.t_L = T.scalar('L')

        self.grad_updates = fista_updates(
            self.t_A, self.t_E_rec, self.t_LAMBDA,
            self.t_L, pos_only=pos_only)

        _, self.t_fista_X, self.t_T = self.grad_updates.keys()

        # Initialize t_A, and extra variables

        inputs = [self.t_XR, self.t_YR, self.t_R, self.t_Wbt, self.t_L]
        self.img_grad = theano.function(inputs=inputs,
                                        outputs=energy_outputs,
                                        updates=self.grad_updates)

        # Define Hessian:
        t_Dp = self.t_D.reshape((self.n_l, self.l_i, self.l_i))  # k, i2, i1

        # b, i2, i1, j, t -> b, _, i2, i1, j, t
        # k, i2, i1 -> _, k, i2, i1, _, _
        # b, k, j, t
        t_SpRFCoupling = (
            t_PixRFCoupling.dimshuffle(0, 'x', 1, 2, 3, 4) *
            t_Dp.dimshuffle('x', 0, 1, 2, 'x', 'x')).sum(axis=(2, 3))

        # b, k, j, t
        t_dlogFPdA = dlogfp_dA(
            t_SpRFCoupling, self.t_G, self.t_IE, self.t_L0, self.t_L1)

        # b, k, k', j, t -> k, k'
        t_E_R_AA = (
            self.t_Wbt.dimshuffle(0, 'x', 'x', 'x', 1) *
            t_dlogFPdA.dimshuffle(0, 'x', 1, 2, 3) *
            t_dlogFPdA.dimshuffle(0, 1, 'x', 2, 3) *
            self.t_FP.dimshuffle(0, 'x', 'x', 1, 2)
            ).sum(axis=(0, 3, 4))

        self.hessian_func = theano.function(
            inputs=[self.t_XR, self.t_YR, self.t_Wbt],
            outputs=t_E_R_AA)

        # Define variables for online learning

        self.t_Ap = theano.shared(A0, 'Ap')  # Previous value of A
        self.t_H = theano.shared(H0, 'H')  # Previous value of hessian

        t_E_prev = (
            0.5 *
            (self.t_A - self.t_Ap).dimshuffle('x', 0) *
            self.t_H *
            (self.t_A - self.t_Ap).dimshuffle(0, 'x')
            ).sum()

    def set_gain_factor(self, g):
        self.t_G.set_value(g)

    def get_A(self):
        return self.t_A.get_value()

    def reset_image_estimate(self):
        """
        Resets the value of the image as stored on the GPU
        """
        self.t_A.set_value(
            np.zeros_like(self.t_A.get_value()).astype('float32'))

    def calculate_L(self, n_t, n_n, l0, l1, dt, d_scl, ctant):
        """
        Return the value of the Lipschitz constant of the smooth part
            of the cost function

        Parameters
        ----------
        n_t : int
            number of timesteps
        n_n : int
            number of neurons
        l0 : float
            baseline firing rate
        l1 : float
            maximum firing rate
        dt : float
            timestep
        d_scl : float
            scale of dictionary elements (mean sum of squares)
        ctant : float
            constant to loosen our bound
        """
        return (n_t * n_n * l1 * dt * d_scl ** 2 *
                np.log(l1 / l0) ** 2 * ctant).astype('float32')

    def reset_m_aux(self):
        """
        Resets auxillary gradient descent variables for the M step
            eg. fista we reset the copy of A and the step size
        """
        self.t_T.set_value(np.array([1.]).astype(theano.config.floatX))
        self.t_fista_X.set_value(self.t_A.get_value())
