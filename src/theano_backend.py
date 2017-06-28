# theano_backend.py

import numpy as np
import theano
import theano.tensor as T
from sparse_coder.fista import fista_updates
from utils.theano_gradient_routines import ada_delta


def reset_shared_var(t_S):
    """
    Reset the value of a theano shared variable to all zeros
    t_S - shared theano variable
    """
    t_S.set_value(np.zeros_like(t_S.get_value()).astype('float32'))


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


def firing_prob(t_Ips, t_G, t_IE, t_L0, t_L1, t_SMIN, t_SMAX, t_DT):
    # Firing probabilities indexed by b, j, t
    # t_Ips - Image-RF inner products indexed as b, j, t
    # t_G - gain constant
    # t_IE - identity of retinal ganglion cells
    # t_L0, t_L1 - min, max firing rate
    # t_DT - time step size

    t_IEr = t_IE.dimshuffle('x', 0, 'x')
    t_Gen = t_IEr + (1 - 2 * t_IEr) * (
        (t_G * t_Ips - t_SMIN) / (t_SMAX - t_SMIN))  # Generator signal
    t_FP = t_DT * T.exp(T.log(t_L0) + T.log(t_L1 / t_L0) * t_Gen)
    return t_FP


def dlogfp_dA(t_dIpsdA, t_G, t_IE, t_L0, t_L1, t_SMIN, t_SMAX):
    """
    t_dIpsdA - d Ips / dA indexed as b, k, j, t
    t_G - gain constant
    t_IE - RGC identity, j
    t_L0, t_L1 - min, max firing rates
    Returns the d log FP / dA indexed b, k, j, t

    """
    t_IEr = t_IE.dimshuffle('x', 'x', 0, 'x')
    t_dGen_dA = (1 - 2 * t_IEr) * t_G * t_dIpsdA / (t_SMAX - t_SMIN)
    t_dlogFPdA = T.log(t_L1 / t_L0) * t_dGen_dA
    return t_dlogFPdA


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


class TheanoBackend(object):

    """
    Theano backend for executing the computations
    """

    def __init__(
        self,
        XS,
        YS,
        XE,
        YE,
        IE,
        Var,
        d,
        l0,
        l1,
        DT,
        G,
        GAMMA,
        LAMBDA,
        TAU,
        QUAD_REG,
        QUAD_REG_MEAN,
        pos_only=True,
        SMIN=0.,
        SMAX=1.
    ):
        """
        Initializes all theano variables and functions
        """
        self.l_i = XS.shape[0]
        self.n_l = d.shape[0]
        self.n_n = XE.shape[0]

        # Define Theano Variables Common to Generation and Inference
        self.t_XS = theano.shared(XS, 'XS')
        self.t_YS = theano.shared(YS, 'YS')
        self.t_XE = theano.shared(XE, 'XE')
        self.t_YE = theano.shared(YE, 'YE')
        self.t_IE = theano.shared(IE, 'IE')
        self.t_Var = theano.shared(Var, 'Var')

        self.t_XR = T.matrix('XR')
        self.t_YR = T.matrix('YR')

        #  Parameters
        self.t_L0 = theano.shared(np.float32(l0), 'L0')
        self.t_L1 = theano.shared(np.float32(l1), 'L1')
        self.t_DT = theano.shared(np.float32(DT), 'DT')
        self.t_G = theano.shared(np.float32(G), 'G')
        self.t_TAU = theano.shared(np.float32(TAU), 'TAU')
        self.t_SMIN = theano.shared(np.float32(SMIN), 'SMIN')
        self.t_SMAX = theano.shared(np.float32(SMAX), 'SMAX')

        ##############################
        # Simulated Spike Generation #
        ##############################

        self.t_S_gen = T.matrix('S_gen')  # Image dims are i2, i1
        self.t_Ips_gen, _ = inner_products(self.t_S_gen, self.t_Var,
                                           self.t_XS, self.t_YS,
                                           self.t_XE, self.t_YE,
                                           self.t_XR, self.t_YR)
        self.t_FP_gen = firing_prob(self.t_Ips_gen, self.t_G, self.t_IE,
                                    self.t_L0, self.t_L1, self.t_SMIN,
                                    self.t_SMAX, self.t_DT)

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

        self.t_R = T.matrix('R')

        # Current value of A
        self.t_A = theano.shared(
            np.zeros((self.n_l,)).astype('float32'), 'A')
        # Previous value of A
        self.t_Ap = theano.shared(
            np.zeros((self.n_l,)).astype('float32'), 'Ap')

        self.t_D = theano.shared(d, 'D')  # Dictionary

        self.t_Wbt = T.matrix('Wbt')  # Weights (b,t) from particle filter

        # Sum of Hessians
        self.t_H = theano.shared(
            np.zeros((self.n_l, self.n_l)).astype('float32'), 'H')
        self.t_B = theano.shared(
            np.zeros((self.n_l,)).astype('float32'), 'B')  # Prior Bias

        # Constants

        self.t_GAMMA = theano.shared(np.float32(GAMMA), 'GAMMA')
        self.t_LAMBDA = theano.shared(np.float32(LAMBDA), 'LAMBDA')
        self.t_QUAD_REG = theano.shared(np.float32(QUAD_REG), 'QUAD_REG')
        self.t_QUAD_REG_MEAN = theano.shared(
            np.float32(QUAD_REG_MEAN), 'QUAD_REG_MEAN')

        # Calculate Firing rate
        self.t_S = T.dot(self.t_A, self.t_D).reshape((self.l_i, self.l_i))
        self.image_est = theano.function(inputs=[], outputs=self.t_S)

        self.t_Ips, t_PixRFCoupling = inner_products(
            self.t_S, self.t_Var, self.t_XS, self.t_YS,
            self.t_XE, self.t_YE, self.t_XR, self.t_YR)

        self.t_FP = firing_prob(self.t_Ips, self.t_G, self.t_IE,
                                self.t_L0, self.t_L1,
                                self.t_SMIN, self.t_SMAX, self.t_DT)

        # Define Hessian
        # Reshape dictionary for computing derivative: k, i2, i1
        t_Dp = self.t_D.reshape((self.n_l, self.l_i, self.l_i))

        # Compute dc/dA = dc/dS * ds/dA
        # b, i2, i1, j, t -> b, _, i2, i1, j, t
        # k, i2, i1 -> _, k, i2, i1, _, _
        # b, k, j, t

        #  t_SpRFCoupling1 = (
        #      t_PixRFCoupling.dimshuffle(0, 'x', 1, 2, 3, 4) *
        #      t_Dp.dimshuffle('x', 0, 1, 2, 'x', 'x')).sum(axis=(2, 3))

        def pix_rf_to_sp_rf(t_PixRFCoupling, t_Dp):
            """
            b i2 i1 j t
            k i2 i1
            b k j t
            """

            tmp1 = t_PixRFCoupling.dimshuffle(1, 2, 0, 3, 4).reshape(
                (self.l_i ** 2, -1))
            # i2i1 bjt

            tmp2 = t_Dp.reshape((self.n_l, -1))  # k i2i1

            tmp3 = T.dot(tmp2, tmp1)  # k bjt
            n_b, n_t = self.t_Wbt.shape
            return tmp3.reshape(
                (self.n_l, n_b, self.n_n, n_t)).dimshuffle(
                    1, 0, 2, 3)

        t_SpRFCoupling = pix_rf_to_sp_rf(t_PixRFCoupling, t_Dp)

        #  self.sp_rf_test= theano.function(
        #      inputs=[self.t_XR, self.t_YR, self.t_Wbt],
        #      outputs=[t_SpRFCoupling, t_SpRFCoupling1])

        # Get RGC Sparse Coeff couplings
        # bkjt,bt-> kj
        t_SpRGCCoupling = (self.t_Wbt.dimshuffle(0, 'x', 'x', 1) *
                           t_SpRFCoupling).sum(axis=(0, 3))

        self.get_sp_rf_coupling = theano.function(
            inputs=[self.t_XR, self.t_YR, self.t_Wbt],
            outputs=t_SpRGCCoupling)

        # b, k, j, t
        t_dlogFPdA = dlogfp_dA(
            t_SpRFCoupling, self.t_G, self.t_IE, self.t_L0, self.t_L1,
            self.t_SMIN, self.t_SMAX)

        #  b, k, k', j, t -> k, k'
        t_dE_R_dAA1 = (
            self.t_Wbt.dimshuffle(0, 'x', 'x', 'x', 1) *
            t_dlogFPdA.dimshuffle(0, 'x', 1, 2, 3) *
            t_dlogFPdA.dimshuffle(0, 1, 'x', 2, 3) *
            self.t_FP.dimshuffle(0, 'x', 'x', 1, 2)
        ).sum(axis=(0, 3, 4))

        def calc_hessian(t_Wbt, t_dlogFPdA, t_FP):
            """
            Calculate the hessian given the following

            Parameters
            ----------
            t_Wbt : theano.tensor, shape (b, t)
            t_dlogFPdA : theano.tensor, shape (b,k,j,t)
            t_FP : theano.tensor, shape (b, j, t)

            Returns
            -------
            t_dE_R_dAA : theano.tensor, shape (k, k')
            """

            tmp = t_Wbt.dimshuffle(0, 'x', 1) * t_FP  # b, j, t
            tmp1 = tmp.dimshuffle(0, 'x', 1, 2) * t_dlogFPdA

            return T.dot(
                tmp1.dimshuffle(1, 0, 2, 3).reshape((self.n_l, -1)),
                t_dlogFPdA.dimshuffle(1, 0, 2, 3).reshape((self.n_l, -1)).T
            )

        t_dE_R_dAA = calc_hessian(self.t_Wbt, t_dlogFPdA, self.t_FP)

        self.sp_rf_test = theano.function(
            inputs=[self.t_XR, self.t_YR, self.t_Wbt],
            outputs=[t_dE_R_dAA, t_dE_R_dAA1])

        self.t_dlogFPdA = t_dlogFPdA
        self.t_dE_R_dAA = t_dE_R_dAA

        # Compute Energy Functions (negative log-likelihood) to minimize

        # Spiking cost separated by b, j, t
        self.t_E_R_f = spiking_cost(self.t_R, self.t_FP)

        self.t_E_R = T.sum(T.sum(self.t_E_R_f, axis=1) * self.t_Wbt)
        self.t_E_R.name = 'E_R'

        self.t_E_bound = self.t_GAMMA * (
            T.sum(T.switch(self.t_S < self.t_SMIN,
                           -(self.t_S - self.t_SMIN), 0.)) +
            T.sum(T.switch(self.t_S > self.t_SMAX,
                           self.t_S - self.t_SMAX, 0.)))
        self.t_E_bound.name = 'E_bound'

        self.t_E_sp = self.t_LAMBDA * T.sum(T.abs_(self.t_A))
        self.t_E_sp.name = 'E_sp'

        #  self.t_E_quad = 0.5 * T.sum(self.t_QUAD_REG *
        #                              ((self.t_A - self.t_QUAD_REG_MEAN) ** 2))
        #  self.t_E_quad.name = 'E_quad'

        # Define bias term
        t_dPrior = T.grad(self.t_E_sp, self.t_A)

        self.t_E_prev = (
            (self.t_A - self.t_Ap).dimshuffle('x', 0) *
            self.t_H *
            (self.t_A - self.t_Ap).dimshuffle(0, 'x')
        ).sum() * 0.5

        self.t_E_lin_prior = ((self.t_A - self.t_Ap) * self.t_B).sum()

        # Split off terms that will go into fista (i.e. not icluding E_sp)
        self.t_E_rec = (
            self.t_E_prev + self.t_E_R +
            self.t_E_lin_prior + self.t_E_bound
            #  + self.t_E_quad
        )
        self.t_E_rec.name = 'E_rec'

        self.t_E = self.t_E_rec + self.t_E_sp
        self.t_E.name = 'E'

        # Cost from poisson terms separated by batches for particle filter log
        # probability
        self.t_E_R_pf = T.sum(self.t_E_R_f, axis=(1, 2))
        self.spike_energy = theano.function(
            inputs=[self.t_XR, self.t_YR, self.t_R],
            outputs=self.t_E_R_pf)

        # Generate costs given a path, spikes, and time-batch weights
        energy_outputs = [
            self.t_E,
            self.t_E_prev,
            self.t_E_R,
            self.t_E_bound,
            self.t_E_sp,
            self.t_E_lin_prior,
            #  self.t_E_quad,
        ]

        self.costs = theano.function(
            inputs=[self.t_XR, self.t_YR, self.t_R, self.t_Wbt],
            outputs=energy_outputs)

        self.image_costs = theano.function(
            inputs=[self.t_XR, self.t_YR, self.t_R,
                    self.t_Wbt, self.t_S],
            outputs=self.t_E_R)

        # Define variables for FISTA minimization
        self.t_L = T.scalar('L')

        self.grad_updates = fista_updates(
            self.t_A, self.t_E_rec, self.t_LAMBDA,
            self.t_L, pos_only=pos_only)

        _, self.t_fista_X, self.t_T = self.grad_updates.keys()

        # Initialize t_A, and extra variables

        inputs = [self.t_XR, self.t_YR, self.t_R, self.t_Wbt, self.t_L]
        self.run_fista_step = theano.function(
            inputs=inputs, outputs=energy_outputs,
            updates=self.grad_updates)

        # Define functions for online learning #

        self.hessian_func = theano.function(
            inputs=[self.t_XR, self.t_YR, self.t_Wbt],
            outputs=t_dE_R_dAA)

        # After each iteration, replace value of Ap with A
        self.update_Ap = theano.function(
            inputs=[], updates=[(self.t_Ap, self.t_A)])

        t_decay = T.exp(- self.t_DT / self.t_TAU *
                        self.t_XR.shape[1].astype('float32'))

        self.update_HB = theano.function(
            inputs=[self.t_XR, self.t_YR, self.t_Wbt],
            updates=[
                (self.t_H, t_decay * self.t_H + t_dE_R_dAA),
                (self.t_B, t_dPrior)])

        # Code for no motion optimizer
        self.t_E_R_no_mo = T.sum(spiking_cost(self.t_R, self.t_FP))
        self.t_E_R_no_mo.name = 'E_R_no_mo'

        t_E_no_mo = self.t_E_R_no_mo + self.t_E_bound
        t_E_no_mo.name = 'E_no_mo'

        t_Rho = T.scalar('Rho')
        t_Eps = T.scalar('Eps')
        ada_updates = ada_delta(t_E_no_mo, self.t_A, *(t_Rho, t_Eps))
        t_ada_Eg2, t_ada_dA2, _ = ada_updates.keys()

        def reset_adadelta_variables(t_A=self.t_A):
            """
            Resets ADA Delta auxillary variables
            """
            A0 = np.zeros_like(t_A.get_value()).astype(theano.config.floatX)
            t_ada_Eg2.set_value(A0)
            t_ada_dA2.set_value(A0)
            t_A.set_value(A0)

        self.reset_adadelta_variables = reset_adadelta_variables

        self.run_image_max_step = theano.function(
            inputs=[self.t_XR, self.t_YR, self.t_R, t_Rho, t_Eps],
            updates=ada_updates,
            outputs=[t_E_no_mo]
        )

    def reset_hessian_and_bias(self):
        """
        Reset the values of the hessian term and the bias term to
            zero to reset the
        """
        #  reset_shared_var(self.t_H)
        self.t_H.set_value(np.diag(self.t_QUAD_REG.get_value()))

        reset_shared_var(self.t_B)

    def set_gain_factor(self, g):
        self.t_G.set_value(g)

    def get_A(self):
        return self.t_A.get_value()

    def reset(self):
        """
        Resets all shared variables changed during the optimization
        """
        self.reset_image_estimate()
        self.init_m_aux()
        self.reset_hessian_and_bias()
        self.reset_adadelta_variables()

    def reset_image_estimate(self):
        """
        Resets the value of the image as stored on the GPU
        """
        #  reset_shared_var(self.t_A)
        self.t_A.set_value(self.t_QUAD_REG_MEAN.get_value())
        reset_shared_var(self.t_Ap)

    def calculate_L(self, n_t, n_n, l0, l1, dt, ctant):
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
        ctant : float
            constant to loosen our bound
        """
        d_scl = np.sqrt((self.t_D.get_value() ** 2).sum(1).mean())

        a = n_t * n_n * l1 * dt * d_scl ** 2 * np.log(l1 / l0) ** 2
        b = 0.5 * self.t_QUAD_REG.get_value().max()
        #  print a, b, b / a
        return ((a + b) * ctant).astype('float32')

    def init_m_aux(self):
        """
        Resets auxillary gradient descent variables for the M step
            eg. fista we reset the copy of A and the step size
        """
        self.t_T.set_value(np.array([1.]).astype(theano.config.floatX))
        self.t_fista_X.set_value(self.t_A.get_value())
