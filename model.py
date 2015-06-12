# Python script containing a class that does expectation maximization
#   to estimate an image from simulated LGN cell responses
# See the end of the script for a sample usage

import numpy as np
import theano
import theano.tensor as T
import os
from scipy.signal import convolve2d
from utils.bounded_diffusion import Center
import utils.particle_filter_new as PF
from utils.theano_gradient_routines import ada_delta
from utils.image_gen import ImageGenerator
from utils.SNR import SNR
import matplotlib.pyplot as plt
import cPickle as pkl
from scipy.io import savemat
from utils.time_filename import time_string
from collections import OrderedDict
from scipy.io import loadmat
from utils.BurakPoissonLP import PoissonLP


class EMBurak:
    def __init__(self, S_gen, D, DT = 0.001, DC = 100., N_T = 50,
                 L_N = 14, a = 1., LAMBDA = 1., save_mode = False):
        """
        Initializes the parts of the EM algorithm
            -- Sets all parameters
            -- Initializes Dictionary
            -- Compiles the theano backend
            -- Sets the gain factor for the spikes
            -- Initializes the object that generates the paths
            -- Initializes the Particle Filter object
            -- Checks that the output directory exists

        S_gen - image that generates the spikes - (L_I, L_I), type = float32
        D - dictionary used to infer latent factors, 
               shape = (N_L, N_pix), type = float32
        Checks for consistency L_I ** 2 = N_pix
        """
        
        self.save_mode = save_mode # If true, save results of each EM iteration
        print 'The save mode is ' + str(save_mode)
        self.D = D.astype('float32') # Dictionary
        self.N_L, self.N_Pix = D.shape
        # N_L - number of latent factors
        # N_pix - number of pixels in the image


        self.S_gen = S_gen.astype('float32')
        self.L_I = S_gen.shape[0] # Linear dimension of the image

        if not self.L_I ** 2 == self.N_Pix:
            raise ValueError('Mismatch between dictionary and image size')

        # Simulation Parameters
        self.DT = DT # Simulation timestep
        self.DC = DC  # Diffusion Constant
        self.L0 = 10.
        self.L1 = 100.
                
        # Problem Dimensions
        self.N_T = N_T # Number of time steps
        self.L_N = L_N # Linear dimension of neuron receptive field grid
        self.N_N = self.L_N ** 2 # Number of neurons
#       self.N_L Number of latent sparse factors
#       self.L_I Linear dimension of image

        # Image Prior Parameters
        self.GAMMA = 100. # Pixel out of bounds cost parameter
        self.LAMBDA = LAMBDA # the sparse prior is delta (S-DA) + LAMBDA * |A|

        # EM Parameters
        # M - Parameters (ADADELTA)
        self.Rho = 0.4
        self.Eps = 0.001
        self.N_g_itr = 10
        self.N_itr = 10

        # E Parameters (Particle Filter)
        self.N_P = 25 # Number of particles for the EM
        
        # Initialize pixel and LGN positions
        # Position of pixels
        self.XS = np.arange(- self.L_I / 2, self.L_I / 2)
        self.YS = np.arange(- self.L_I / 2, self.L_I / 2)
        self.XS = self.XS.astype('float32')
        self.YS = self.YS.astype('float32') 

        # Position of LGN receptive fields
        self.a = a # Receptive field spacing
        
        self.XE, self.YE = np.meshgrid(np.arange(- self.L_N / 2, self.L_N / 2),
                                       np.arange(- self.L_N / 2, self.L_N / 2))

        self.XE = self.XE.ravel().astype('float32') 
        self.YE = self.YE.ravel().astype('float32') 
        self.XE = self.XE * self.a
        self.YE = self.YE * self.a
        

        # Variances of gaussians for each pixel
        self.Var = 0.25 * np.ones((self.L_I,)).astype('float32') 
        # Gain factor (to be set later)
        self.G = 1. 

        # X-Position of retina (batches, timesteps), batches used for inference only
        self.XR = np.zeros((1, self.N_T)).astype('float32') 
        # Y-Position of retina
        self.YR = np.zeros((1, self.N_T)).astype('float32') 
        # Spikes (1 or 0)
        self.R = np.zeros((self.N_N, self.N_T)).astype('float32')  

        # Sparse Coefficients
        self.A = np.zeros((self.N_L,)).astype('float32')      


        # Shapes of other variables used elsewhere

        # Pixel values for generating image (same shape as estimated image)
#        self.S_gen = np.zeros((self.L_I, self.L_I)).astype('float32') 
        # Assumes that the first dimension is 'Y' 
        #    and the second dimension is 'X'

        
        # Weighting for batches and time points
        #self.Wbt = np.ones((self.N_B, self.N_T)).astype('float32') 
        
        # Dictionary going from latent factors to image
#       self.D = np.zeros((self.N_L, self.N_Pix)).astype('float32')  

            

        self.init_theano_core()
        self.set_gain_factor()
        self.init_path_generator()
        
        if (self.save_mode):
            self.init_output_dir()
        

    def gen_data(self):
        """
        Generates a path and spikes
        Builds a dictionary saving these data
        """
        self.gen_path()
        self.gen_spikes()
        self.build_param_and_data_dict()
        self.init_particle_filter()

             
    def run(self):
        """
        Runs an iteration of EM
        Then saves relevant summary information about the run
        """
        self.run_EM()
        if (self.save_mode):
            self.save()


    def init_theano_core(self):
        """
        Initializes all theano variables and functions
        """
        # Define Theano Variables Common to Generation and Inference
        self.t_XS = theano.shared(self.XS, 'XS')
        self.t_YS = theano.shared(self.YS, 'YS')
        self.t_XE = theano.shared(self.XE, 'XE')
        self.t_YE = theano.shared(self.YE, 'YE')
        self.t_Var = theano.shared(self.Var, 'Var')

        self.t_XR = T.matrix('XR')
        self.t_YR = T.matrix('YR')
        self.t_R = T.matrix('R')

        #  Parameters
        self.t_L0 = T.scalar('L0')
        self.t_L1 = T.scalar('L1')
        self.t_DT = T.scalar('DT')
        self.t_DC = T.scalar('DC')
        self.t_G = T.scalar('G')

        def inner_products(t_S, t_Var, t_XS, t_YS, t_XE, t_YE, t_XR, t_YR):
            """
            Take dot product of image with an array of gaussians
            t_S - theano image variable dimensions i2, i1
            t_Var - variances of receptive fields
            t_XS - X coordinate for image pixels for dimension i1
            t_YS - Y coordinate for image pixels for dimension i2
            t_XE - X coordinate for receptive fields j
            t_YE - Y coordinate for receptive fields j
            t_XR - X coordinate for retina in form batch, timestep, b,t
            t_YR - Y coordinate ''
            """
            
            # Note in this computation, we do the indices in this form:
            #  b, i, j, t
            #  batch, pixel, neuron, timestep

            # indices: b, i1, j, t
            t_dX = (t_XS.dimshuffle('x', 0, 'x', 'x') 
                     - t_XE.dimshuffle('x', 'x', 0, 'x') 
                     - t_XR.dimshuffle(0, 'x', 'x', 1))
            t_dX.name = 'dX'
            # indices: b, i2, j, t
            t_dY = (t_YS.dimshuffle('x', 0, 'x', 'x') 
                     - t_YE.dimshuffle('x', 'x', 0, 'x') 
                     - t_YR.dimshuffle(0, 'x', 'x', 1))
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
                            t_PixRFCouplingY.dimshuffle(0, 1, 'x', 2, 3), axis = 1)
            # Sum_i1 T(b, i1, j, t) * T(b, i2, j, t) = T(b, j, t)
            t_Ips = T.sum(t_IpsY * t_PixRFCouplingX, axis = 1)
            t_Ips.name = 'Ips'
            
            # For the gradient, we also prepare d Ips / dS
            # This is in the form b, i2, i1, j, t
            t_PixRFCoupling = (t_PixRFCouplingX.dimshuffle(0, 'x', 1, 2, 3) * 
                               t_PixRFCouplingY.dimshuffle(0, 1, 'x', 2, 3))

            return t_Ips, t_PixRFCoupling

        self.inner_products = inner_products
      
        def firing_prob(t_Ips, t_G, t_L0, t_L1, t_DT):
            # Firing probabilities indexed by b, j, t
            # t_Ips - Image-RF inner products indexed as b, j, t
            # t_G - gain constant
            # t_L0, t_L1 - min, max firing rate
            # t_DT - time step size

            t_FP_0 = t_DT * T.exp(T.log(t_L0) + T.log(t_L1 / t_L0) * t_G * t_Ips)

            t_FP = T.switch(t_FP_0 > 0.9, 0.9, t_FP_0)
            return t_FP
        
        self.firing_prob = firing_prob

        # Simulated Spike Generation
        
        self.t_S_gen = T.matrix('S_gen') # Image dims are i2, i1
        self.t_Ips_gen, _ = inner_products(self.t_S_gen, self.t_Var,
                                        self.t_XS, self.t_YS,
                                        self.t_XE, self.t_YE,
                                        self.t_XR, self.t_YR)
        self.t_FP_gen = firing_prob(self.t_Ips_gen, self.t_G, 
                                    self.t_L0, self.t_L1, self.t_DT)

        # Computes image-RF inner products and the resulting firing probabilities
        self.RFS = theano.function(inputs = [self.t_S_gen, self.t_XR, self.t_YR,
                                             self.t_L0, self.t_L1, 
                                             self.t_DT, self.t_G],
                                   outputs = [self.t_Ips_gen, self.t_FP_gen])
        self.rng = T.shared_randomstreams.RandomStreams(seed = 10)
        self.t_R_gen = (self.rng.uniform(size = self.t_FP_gen.shape) 
                        < self.t_FP_gen).astype('float32')


        self.spikes = theano.function(inputs = [self.t_S_gen, self.t_XR, self.t_YR,
                                                self.t_L0, self.t_L1, 
                                                self.t_DT, self.t_G],
                                      outputs = self.t_R_gen)
        
        # Latent Variable Estimation

        def spiking_cost(t_R, t_FP):
            """
            Returns the negative log likelihood of the spikes given the inner products
            t_R - spikes in form j, t
            t_FP - Firing probabilities in form b, j, t
            Returns -log p(R|X,S) with indices in the form b, j, t
            """
            t_E_R_f = -(t_R.dimshuffle('x', 0, 1) * T.log(t_FP) 
                     + (1 - t_R.dimshuffle('x', 0, 1)) * T.log(1 - t_FP))
            t_E_R_f.name = 'E_R_f'
            return t_E_R_f

        self.spiking_cost = spiking_cost

        self.t_A = theano.shared(self.A, 'A')
        self.t_D = theano.shared(self.D, 'D')
        self.t_S = T.dot(self.t_A, self.t_D).reshape((self.L_I, self.L_I))
        self.image_est = theano.function(inputs = [], outputs = self.t_S)
        # FIXME: shouldn't access L_I, Image dims are i2, i1

        self.t_GAMMA = T.scalar('GAMMA')
        self.t_LAMBDA = T.scalar('LAMBDA')

        self.t_Ips, _ = inner_products(self.t_S, self.t_Var,
                                       self.t_XS, self.t_YS,
                                       self.t_XE, self.t_YE,
                                       self.t_XR, self.t_YR)

        self.t_FP = firing_prob(self.t_Ips, self.t_G, 
                                self.t_L0, self.t_L1, self.t_DT)

        # Compute Energy Functions (negative log-likelihood) to minimize
        self.t_Wbt = T.matrix('Wbt') # Weights (batch, timestep) from particle filter
        self.t_E_R_f = spiking_cost(self.t_R, self.t_FP)

        self.t_E_R = T.sum(T.sum(self.t_E_R_f, axis = 1)  * self.t_Wbt)
        self.t_E_R.name = 'E_R'

        self.t_E_bound = self.t_GAMMA * (
                        T.sum(T.switch(self.t_S < 0., -self.t_S, 0)) + 
                        T.sum(T.switch(self.t_S > 1., self.t_S - 1, 0)))
        self.t_E_bound.name = 'E_bound'
                    

        self.t_E_sp = self.t_LAMBDA * T.sum(T.abs_(self.t_A))
        self.t_E_sp.name = 'E_sp'
        
        self.t_E = self.t_E_R + self.t_E_bound + self.t_E_sp
        self.t_E.name = 'E'

        # Cost from poisson terms separated by batches for particle filter log probability
        self.t_E_R_b = T.sum(self.t_E_R_f, axis = (1, 2))
        self.spike_energy = theano.function(inputs = [self.t_XR, self.t_YR, 
                                                      self.t_R,
                                                      self.t_L0, self.t_L1, 
                                                      self.t_DT, self.t_G],
                                            outputs = self.t_E_R_b)

        # Generate costs given a path, spikes, and time-batch weights        
        energy_outputs = [self.t_E, self.t_E_bound, self.t_E_R, self.t_E_sp]          
        self.costs = theano.function(inputs = [self.t_XR, self.t_YR, self.t_R, self.t_Wbt,
                                               self.t_L0, self.t_L1, self.t_DT, self.t_G, 
                                               self.t_GAMMA, self.t_LAMBDA], 
                                     outputs = energy_outputs)


        # Define theano variables for gradient descent
        self.t_Rho = T.scalar('Rho')
        self.t_Eps = T.scalar('Eps')
        self.t_ada_params = (self.t_Rho, self.t_Eps)

        self.grad_updates = ada_delta(self.t_E, self.t_A, *self.t_ada_params)
        self.t_A_Eg2, self.t_A_EdS2, _ = self.grad_updates.keys()
        
        inputs = [self.t_XR, self.t_YR, self.t_R, self.t_Wbt,
                  self.t_L0, self.t_L1, self.t_DT, self.t_G, 
                  self.t_GAMMA, self.t_LAMBDA, self.t_Rho, self.t_Eps]
        
        self.img_grad = theano.function(inputs = inputs,
                                        outputs = energy_outputs,
                                        updates = self.grad_updates)

    def set_gain_factor(self):
        """
        Sets the gain factor so that an image with pixels of intensity 1
            results in spikes at the maximum firing rate
        """
        self.G = 1.
        Ips, FP = self.RFS(np.ones_like(self.S_gen), self.XR, self.YR, 
                           self.L0, self.L1, 
                           self.DT, self.G)
        self.G = (1. / Ips.max()).astype('float32')

    def init_particle_filter(self):
        """
        Initializes the particle filter class
        Requires spikes to already be generated
        """
        # Define necessary components for the particle filter
        D_H = 2 # Dimension of hidden state (i.e. x,y = 2 dims)
        sdev = np.sqrt(self.DC * self.DT / 2) # Needs to be sdev per component
        ipd = PF.GaussIPD(D_H, self.N_N, sdev * 0.001)
        tpd = PF.GaussTPD(D_H, self.N_N, sdev)
        ip = PF.GaussIP(D_H, sdev * 0.001)
        tp = PF.GaussTP(D_H, sdev)
        lp = PoissonLP(self.N_N, self.L0, self.L1, 
                       self.DT, self.G, self.spike_energy)
        self.pf = PF.ParticleFilter(ipd, tpd, ip, tp, lp, 
                                    self.R.transpose(), self.N_P)

    def init_output_dir(self):
        """
        Create an output directory 
        """
        if not os.path.exists('output'):
            os.mkdir('output')
            
        self.output_dir = 'output/' + time_string()
        
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            

    def init_path_generator(self):
        """
        Initialize the path generator
        """
        self.c = Center(self.L_I, self.DC, self.DT)

        
    def gen_path(self):
        """
        Generate a retinal path. Note that the path has a bias towards the
            center so that the image does not go too far out of range
        """
        self.c.reset()
        for t in range(self.N_T):
            x = self.c.get_center()
            self.XR[0, t] = x[0]
            self.YR[0, t] = x[1]
            self.c.advance()


    def gen_spikes(self):
        """
        Generate LGN responses given the path and the image
        """
        self.R = self.spikes(self.S_gen, self.XR, self.YR, self.L0, 
                             self.L1, self.DT, self.G)[0]
        print 'Mean firing rate ' + str(self.R.mean() / self.DT)


    def true_costs(self):
        """
        Prints out the negative log-likelihood of the observed spikes given the
            image and path that generated them
        Note that the energy is normalized by the number of timesteps
        """
        print 'Pre-EM testing'
        args = (self.XR, self.YR, self.R, self.Wbt, 
                self.L0, self.L1, self.DT, self.G, 
                self.GAMMA, self.LAMBDA)        
        
        out = self.costs(*args)
        
        E, E_bound, E_R, E_rec, E_sp = out
        print 'Costs of underlying data ' + str((E/self.N_T, E_rec/self.N_T))


    def reset_image_estimate(self):
        """
        Resets the value of the image as stored on the GPU
        """
        self.t_A.set_value(np.zeros_like(self.A).astype('float32'))
            
   
    def run_E(self, t):
        """
        Runs the the particle filter until it has run a total of t time steps
        t - number of timesteps
        The result is saved in self.pf.XS,WS,means
        """
        if (t > self.N_T): 
            raise IndexError('Maximum simulated timesteps exceeded in E step')
        while (self.pf.t < t):
            self.pf.advance()
        self.pf.calculate_means_sdevs()
        
        print 'Path SNR ' + str(SNR(self.XR[0][0:t], self.pf.means[0:t, 0]))
    
    
    def reset_M_aux(self):
        """
        Resets auxillary gradient descent variables for the M step
            eg. for ADADelta, we reset the RMS of dx and g
        """
        self.t_A_Eg2.set_value(np.zeros_like(self.A).astype('float32'))
        self.t_A_EdS2.set_value(np.zeros_like(self.A).astype('float32'))
        
        
    
    def run_M(self, t, N_g_itr = 5):
        """
        Runs the maximization step for the first t time steps
        resets the values of auxillary gradient descent variables at start
        t - number of time steps
        result is saved in t_A.get_value()
        """
        self.reset_M_aux()
        print 'Spike Energy / t | Bound. Energy / t | SNR' 
        for v in range(N_g_itr):
            args = (self.pf.XS[0:t, :, 0].transpose(),
                    self.pf.XS[0:t, :, 1].transpose(),
                    self.R[:, 0:t], self.pf.WS[0:t].transpose(),
                    self.L0, self.L1, self.DT, 
                    self.G, self.GAMMA, self.LAMBDA,
                    self.Rho, self.Eps)
                    
            out = self.img_grad(*args)
            self.img_SNR = SNR(self.S_gen, self.image_est())
        
        print self.print_costs(out, t) + str(self.img_SNR)


    def print_costs(self, out, t):
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

        
    def run_EM(self, N_itr = None, N_g_itr = None):
        """
        Runs full expectation maximization algorithm
        N_itr - number of iterations of EM
        N_g_itr - number of gradient steps in M step
        Saves summary of run info in self.data 
        Note running twice will overwrite this run info
        """
        if N_itr != None:
            self.N_itr = N_itr
        if N_g_itr != None:
            self.N_g_itr = N_g_itr
        
        self.reset_image_estimate()
            
        EM_data = {}
    
        print 'Running full EM'
        
        for u in range(self.N_itr):
            t = self.N_T * (u + 1) / self.N_itr
            print ('Iteration number ' + str(u) + 
                   ' Running up time time = ' + str(t))
            
            # Run E step
            self.run_E(t)
            
            # Run M step
            self.run_M(t, N_g_itr = self.N_g_itr)
            
            iteration_data = {}
            iteration_data['time_steps'] = t
            iteration_data['path_means'] = self.pf.means
            iteration_data['path_sdevs'] = self.pf.sdevs
            iteration_data['image_est'] = self.image_est()
            
            iteration_data['coeff_est'] = self.t_A.get_value()
            
            EM_data[u] = iteration_data
            
        self.data['EM_data'] = EM_data

    def build_param_and_data_dict(self):
        """
        Creates a dictionary, self.data, that has all of the parameters of the model
        """
        data = {}
        data['DT'] = self.DT
        data['DC'] = self.DC
        data['L0'] = self.L0
        data['L1'] = self.L1
        data['GAMMA'] = self.GAMMA
        
        data['LAMBDA'] = self.LAMBDA
        data['D'] = self.D
        data['N_L'] = self.N_L
        
        data['a'] = self.a

        data['N_T'] = self.N_T
        data['L_I'] = self.L_I
        data['L_N'] = self.L_N
        
        data['Rho'] = self.Rho
        data['Eps'] = self.Eps

        data['N_g_itr'] = self.N_g_itr
        data['N_itr'] = self.N_itr

        data['N_P'] = self.N_P

        data['XS'] = self.XS
        data['YS'] = self.YS
        data['XE'] = self.XE
        data['YE'] = self.YE
        data['Var'] = self.Var
        data['G'] = self.G

        data['XR'] = self.XR
        data['YR'] = self.YR

        data['S_gen'] = self.S_gen
        
        self.data = data

        
    def save(self):
        """
        Saves information relevant to the EM run
        data.pkl - saves dictionary with all data relevant to EM run
        (Only includes dict for EM data if that was run)
        """
        fn = self.output_dir + '/data_' + time_string() + '.pkl'        
        pkl.dump(self.data, open(fn, 'wb'))


#    def true_path_infer_image_costs(self, N_g_itr = 10):
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
    
   
#    def calculate_inner_products(self):
#        """
#        Calculates the inner products used
#        """
#        self.Ips, self.FP = self.RFS(self.XR, self.YR, 
#                                     self.L0, self.L1, 
#                                     self.DT, self.G)
