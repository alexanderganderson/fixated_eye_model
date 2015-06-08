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
    def __init__(self, DT = 0.002, DC = 40., N_T = 200,
                 L_I = 14, L_N = 18, N_L = 49, a = 1., ALPHA = 0.):
        """
        Initializes the parts of the EM algorithm
            -- Sets all parameters
            -- Initializes Dictionary (if using sparse prior)
            -- Defines relevant theano variables
            -- Compiles theano functions
            -- Sets the gain factor for the spikes
            -- Initializes the Image
            -- Initializes the object that generates the paths
            -- Initializes the Particle Filter object
            -- Checks that the output directory exists
        """
        
        self.debug = False # If True, show debug images
        self.sparse_prior = True # If true, include sparse prior
        self.save_mode = True # If true, save results of each EM iteration
        
        # Simulation Parameters
        self.DT = DT # Simulation timestep
        self.DC = DC  # Diffusion Constant
        self.L0 = 10.
        self.L1 = 100.
        
        # Image Prior Parameters
        self.GAMMA = 100. # Pixel out of bounds cost parameter

        if self.sparse_prior:
            self.ALPHA  = ALPHA # Prior Strength
            self.LAMBDA = 0. # Sparsity constant, set when loading dictionary
            # the sparse prior is ALPHA * ((S-DA) ** 2 + LAMBDA * |A|)
        
        # Problem Dimensions
        if (self.sparse_prior):
            self.N_L = N_L # Number of latent sparse factors
        self.N_T = N_T # Number of time steps
        self.L_I = L_I # Linear dimension of image
        self.L_N = L_N # Linear dimension of neuron receptive field grid

        self.N_B = 1 # Number of batches of data (must be 1)

        # EM Parameters
        # M - Parameters (ADADELTA)
        self.Rho = 0.4
        self.Eps = 0.001
        self.N_g_itr = 10
        self.N_itr = 10

        # E Parameters (Particle Filter)
        self.N_P = 25 # Number of particles for the EM
        
        # Other Parameters
        self.N_Pix = self.L_I ** 2 # Number of pixels in image
        self.N_N = self.L_N ** 2 # Number of neurons
        

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
        
        # Pixel values
        self.S = np.zeros((self.L_I, self.L_I)).astype('float32') 
        # Assumes that the first dimension is 'Y' 
        #    and the second dimension is 'X'

        # Variances of gaussians for each pixel
        self.Var = 0.25 * np.ones((self.L_I,)).astype('float32') 
        # Gain factor set later
        self.G = 1. 

        # X-Position of retina
        self.XR = np.zeros((self.N_B, self.N_T)).astype('float32') 
        # Y-Position of retina
        self.YR = np.zeros((self.N_B, self.N_T)).astype('float32') 
        # Spikes (1 or 0)
        self.R = np.zeros((self.N_N, self.N_T)).astype('float32')  
        
        # Weighting for batches and time points
        self.Wbt = np.ones((self.N_B, self.N_T)).astype('float32') 
        
        if self.sparse_prior:
            # Dictionary going from latent factors to image
            self.D = np.zeros((self.N_L, self.N_Pix)).astype('float32')  
            # Sparse Coefficients
            self.A = np.zeros((self.N_L,)).astype('float32')      
            
        
        
        if self.sparse_prior:
            self.init_dictionary()
            
        self.init_theano_vars()
        self.init_theano_funcs()
        self.set_gain_factor()
        
        self.init_image()
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


    def init_dictionary(self):
        """
        Loads in a dictionary
        Loads the given value of the sparsity penalty
        """
        try:
            data = loadmat('data/mnist_dictionary.mat')
            self.D[:, :] = data['D']
            self.LAMBDA = data['Alpha'].astype('float32')[0, 0]
        except IOError:
            print 'Need to have a dictionary file'
            raise IOError


    def init_theano_vars(self):
        """
        Initializes all theano variables
        """
        # Define Theano Variables
        self.t_XS = theano.shared(self.XS, 'XS')
        self.t_YS = theano.shared(self.YS, 'YS')
        self.t_XE = theano.shared(self.XE, 'XE')
        self.t_YE = theano.shared(self.YE, 'YE')
        self.t_Var = theano.shared(self.Var, 'Sig')

        # dims are i2, i1
        self.t_S = theano.shared(self.S, 'S')

        self.t_XR = T.matrix('XR')
        self.t_YR = T.matrix('YR')
        self.t_R = T.matrix('R')
        if self.sparse_prior:
            self.t_D = theano.shared(self.D, 'D')
            self.t_A = theano.shared(self.A, 'A')

        self.t_Wbt = T.matrix('Wbt')

        # Theano Parameters
        self.t_L0 = T.scalar('L0')
        self.t_L1 = T.scalar('L1')
        self.t_DT = T.scalar('DT')
        self.t_DC = T.scalar('DC')
        self.t_GAMMA = T.scalar('GAMMA')
        if self.sparse_prior:
            self.t_ALPHA = T.scalar('ALPHA')
            self.t_LAMBDA = T.scalar('LAMBDA')
        

        def inner_products(t_S, t_Var, t_XS, t_YS, t_XE, t_YE, t_XR, t_YR):
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
            return t_Ips


        self.t_Ips = inner_products(self.t_S, self.t_Var,
                                    self.t_XS, self.t_YS,
                                    self.t_XE, self.t_YE,
                                    self.t_XR, self.t_YR)

        self.t_G = T.scalar('G')

        # Take dot product of image with an array of gaussians
        
        # Note in this computation, we do the indices in this form:
        #  b, i, j, t
        #  batch, pixel, neuron, timestep
        
        # Firing probabilities indexed by
        # b, j, t
        self.t_FP_0 = self.t_DT * T.exp(T.log(self.t_L0) + 
                                        T.log(self.t_L1 / self.t_L0) * 
                                        self.t_G * self.t_Ips)

        self.t_FP = T.switch(self.t_FP_0 > 0.9, 0.9, self.t_FP_0)

        # Compute Energy Functions (negative log-likelihood) to minimize
        # Note energy is weighted in time and batch by W_bt (used by particle filter)
        self.t_E_R = -T.sum(T.sum(self.t_R.dimshuffle('x', 0, 1) 
                                  * T.log(self.t_FP) 
                                  + (1 - self.t_R.dimshuffle('x', 0, 1)) 
                                  * T.log(1 - self.t_FP), axis = 1) 
                                  * self.t_Wbt)
        self.t_E_R.name = 'E_R'

        self.t_E_bound = self.t_GAMMA * (
                        T.sum(T.switch(self.t_S < 0., -self.t_S, 0)) + 
                        T.sum(T.switch(self.t_S > 1., self.t_S - 1, 0)))
        self.t_E_bound.name = 'E_bound'
                    
        if self.sparse_prior:
            # FIXME: shouldn't access L_I
            self.t_E_rec = self.t_ALPHA * T.sum((self.t_S - 
                                            T.dot(self.t_A, self.t_D).reshape((self.L_I, self.L_I)) ) ** 2)  
            self.t_E_rec.name = 'E_bound'
            
            self.t_E_sp =  (self.t_ALPHA * self.t_LAMBDA 
                            * T.sum(T.abs_(self.t_A)))
            self.t_E_sp.name = 'E_sp'


        self.t_E = self.t_E_R + self.t_E_bound
        
        if self.sparse_prior:
            self.t_E = self.t_E + self.t_E_rec + self.t_E_sp
         
        self.t_E.name = 'E'

        # Auxiliary Theano Variables

        # Cost from poisson terms separated by batches for particle filter log probability
        self.t_E_R_b = -T.sum(self.t_R.dimshuffle('x', 0, 1) 
                              * T.log(self.t_FP) 
                              + (1 - self.t_R.dimshuffle('x', 0, 1)) 
                              * T.log(1 - self.t_FP), axis = (1, 2))

        # Generate Spikes
        self.rng = T.shared_randomstreams.RandomStreams(seed = 10)
        self.t_R_gen = (self.rng.uniform(size = self.t_FP.shape) 
                        < self.t_FP).astype('float32')

    def init_theano_funcs(self):
        # Computes image-RF inner products and the resulting firing probabilities
        self.RFS = theano.function(inputs = [self.t_XR, self.t_YR,
                                             self.t_L0, self.t_L1, 
                                             self.t_DT, self.t_G],
                                   outputs = [self.t_Ips, self.t_FP])


        # Initially generate spikes
        self.spikes = theano.function(inputs = [self.t_XR, self.t_YR,
                                                self.t_L0, self.t_L1, 
                                                self.t_DT, self.t_G],
                                      outputs = self.t_R_gen)
        
        # Generate costs given a path, spikes, and time-batch weights
        
        inputs = [self.t_XR, self.t_YR, self.t_R, self.t_Wbt,
                  self.t_L0, self.t_L1, self.t_DT, self.t_G, 
                  self.t_GAMMA]
        outputs = [self.t_E, self.t_E_bound, self.t_E_R]          
        
        if self.sparse_prior:
            inputs.append(self.t_ALPHA)
            inputs.append(self.t_LAMBDA) 
            outputs.append(self.t_E_rec)
            outputs.append(self.t_E_sp)
            
        self.costs = theano.function(inputs = inputs, outputs = outputs)


        # Returns the energy E_R = -log P(R|X,S) separated by batches
        # Function for the particle filter
        self.spike_energy = theano.function(inputs = [self.t_XR, self.t_YR, 
                                                      self.t_R,
                                                      self.t_L0, self.t_L1, 
                                                      self.t_DT, self.t_G],
                                            outputs = self.t_E_R_b)



        # Define theano variables for gradient descent
        self.t_Rho = T.scalar('Rho')
        self.t_Eps = T.scalar('Eps')
        self.t_ada_params = (self.t_Rho, self.t_Eps)

        self.s_grad_updates = ada_delta(self.t_E, self.t_S, *self.t_ada_params)
        self.t_S_Eg2, self.t_S_EdS2, _ = self.s_grad_updates.keys()
        
        if self.sparse_prior:
            self.a_grad_updates = ada_delta(self.t_E, self.t_A, 
                                            *self.t_ada_params)
            self.t_A_Eg2, self.t_A_EdS2, _ = self.a_grad_updates.keys()
        
        self.grad_updates = OrderedDict()        
        for key in self.s_grad_updates.keys():
            self.grad_updates[key] = self.s_grad_updates[key]

        if self.sparse_prior:
            for key in self.a_grad_updates.keys():
                self.grad_updates[key] = self.a_grad_updates[key]
        
        
        inputs = [self.t_XR, self.t_YR, self.t_R, self.t_Wbt,
                  self.t_L0, self.t_L1, self.t_DT, self.t_G, 
                  self.t_GAMMA, self.t_Rho, self.t_Eps]
        if self.sparse_prior:
            inputs.append(self.t_ALPHA)
            inputs.append(self.t_LAMBDA)
        
        # Use the same outputs as before
        
        self.img_grad = theano.function(inputs = inputs,
                                        outputs = outputs,
                                        updates = self.grad_updates)

        
    def set_gain_factor(self):
        """
        Sets the gain factor so that an image with pixels of intensity 1
            results in spikes at the maximum firing rate
        """
        self.G = 1.
        self.t_S.set_value(np.ones_like(self.S))
        Ips, FP = self.RFS(self.XR, self.YR, 
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


    def init_image(self):
        """
        Initialize the Image
        """
        self.ig = ImageGenerator(self.L_I)
        #self.ig.make_big_E()
        self.ig.make_digit()
        self.ig.normalize()
        self.S = self.ig.img
        self.t_S.set_value(self.S)
        if (self.debug):
            self.ig.plot()

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
        for b in range(self.N_B):
            self.c.reset()
            for t in range(self.N_T):
                x = self.c.get_center()
                self.XR[b, t] = x[0]
                self.YR[b, t] = x[1]
                self.c.advance()


    def plot_path(self):
        """
        Plots the path corresponding to the first batch
        """
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(self.N_T) * self.DT, self.XR[0])
        plt.title('x coordinate')
        plt.subplot(1, 2, 2)
        plt.plot(np.arange(self.N_T) * self.DT, self.YR[0])
        plt.title('y coordinate')
        plt.show()


    def gen_spikes(self):
        """
        Generate LGN responses given the path and the image
        """
        self.t_S.set_value(self.S)
        self.R = self.spikes(self.XR, self.YR, self.L0, 
                             self.L1, self.DT, self.G)[0]
        print 'Mean firing rate ' + str(self.R.mean() / self.DT)


    def true_costs(self):
        """
        Prints out the negative log-likelihood of the observed spikes given the
            image and path that generated them
        Note that the energy is normalized by the number of timesteps
        """
        print 'Pre-EM testing'
        self.t_S.set_value(self.S)
        # FIXME: different number of args for sparse_prior
        args = (self.XR, self.YR, self.R, self.Wbt, 
                self.L0, self.L1, self.DT, self.G, 
                self.GAMMA)
        if self.sparse_prior:
            args = args + (self.ALPHA, self.LAMBDA)        
        
        out = self.costs(*args)
        
        if self.sparse_prior:
            E, E_bound, E_R, E_rec, E_sp = out
        else: 
            E, E_bound, E_R = out        
        print 'Costs of underlying data ' + str((E/self.N_T, E_rec/self.N_T))


    def reset_image_estimate(self):
        """
        Resets the value of the image as stored on the GPU
        """
        self.t_S.set_value(0.5 + np.zeros(self.S.shape).astype('float32'))
        if self.sparse_prior:
            self.t_A.set_value(np.zeros_like(self.A).astype('float32'))
            
   
    def run_E(self, t):
        """
        Runs the the particle filter until it has run a total of t time steps
        t - number of timesteps
        The result is saved in self.pf.XS,WS,means
        """
        if (t > self.N_T): 
            raise IndexError('Maximum simulated timesteps exceeded in E step')
        #self.pf.run(self.R.transpose()[0:t], self.N_P)

        while (self.pf.t < t):
            self.pf.advance()
        self.pf.calculate_means_sdevs()
        
        print 'Path SNR ' + str(SNR(self.XR[0][0:t], self.pf.means[0:t, 0]))
    
    
    def reset_M_aux(self):
        """
        Resets auxillary gradient descent variables for the M step
            eg. for ADADelta, we reset the RMS of dx and g
        """
        self.t_S_Eg2.set_value(np.zeros_like(self.S).astype('float32'))
        self.t_S_EdS2.set_value(np.zeros_like(self.S).astype('float32'))
        
        if self.sparse_prior:
            self.t_A_Eg2.set_value(np.zeros_like(self.A).astype('float32'))
            self.t_A_EdS2.set_value(np.zeros_like(self.A).astype('float32'))
        
        
    
    def run_M(self, t, N_g_itr = 5):
        """
        Runs the maximization step for the first t time steps
        resets the values of auxillary gradient descent variables at start
        t - number of time steps
        result is saved in t_S.get_value()
        """
        self.reset_M_aux()
        print 'Spike Energy / t | Bound. Energy / t | SNR' 
        for v in range(N_g_itr):
            args = (self.pf.XS[0:t, :, 0].transpose(),
                    self.pf.XS[0:t, :, 1].transpose(),
                    self.R[:, 0:t], self.pf.WS[0:t].transpose(),
                    self.L0, self.L1, self.DT, 
                    self.G, self.GAMMA,
                    self.Rho, self.Eps)
            if self.sparse_prior:
                args = args + (self.ALPHA, self.LAMBDA)
            
            out = self.img_grad(*args)
            self.img_SNR = SNR(self.S, self.t_S.get_value())
        
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
            t = self.N_T * (u + 1) / self.N_itr #t = self.N_T
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
            iteration_data['image_est'] = self.t_S.get_value()
            
            if self.sparse_prior:
                iteration_data['coeff_est'] = self.t_A.get_value()
            
            EM_data[u] = iteration_data
            
        self.data['EM_data'] = EM_data

    def build_param_and_data_dict(self):
        """
        Creates a dictionary, self.data, that has all of the parameters of the model
        """
        
        data = {}
        data['sparse_prior'] = self.sparse_prior
        data['DT'] = self.DT
        data['DC'] = self.DC
        data['L0'] = self.L0
        data['L1'] = self.L1
        data['GAMMA'] = self.GAMMA
        
        if self.sparse_prior:
            data['ALPHA'] = self.ALPHA
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

        data['S'] = self.S
        
        self.data = data

        
    def save(self):
        """
        Saves information relevant to the EM run
        data.pkl - saves dictionary with all data relevant to EM run
        (Only includes dict for EM data if that was run)
        """
        fn = self.output_dir + '/data_' + time_string() + '.pkl'        
        pkl.dump(self.data, open(fn, 'wb'))


    def plot_image_estimate(self):
        """
        Plot the estimated image against the actual image
        """
        vmin = -1.
        vmax = 1.
        plt.subplot(1, 3, 1)
        plt.title('Estimate')
        plt.imshow(self.t_S.get_value().reshape(self.L_I, self.L_I), 
                   cmap = plt.cm.gray, 
                   interpolation = 'nearest', 
                   vmin = vmin, vmax = vmax)
        plt.colorbar()
        plt.subplot(1, 3, 2)
        plt.title('Actual')
        plt.imshow(self.S.reshape(self.L_I, self.L_I), cmap = plt.cm.gray, 
                   interpolation = 'nearest',
                   vmin = vmin, vmax = vmax)
        plt.colorbar()
        plt.subplot(1, 3, 3)
        plt.title('Error')
        plt.imshow(np.abs(self.t_S.get_value() 
                          - self.S).reshape(self.L_I, self.L_I), 
                   cmap = plt.cm.gray, interpolation = 'nearest')
        plt.colorbar()
        plt.show()



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
 

if __name__ == '__main__':
    DCs = [0.1]
    for DC in DCs:
        emb = EMBurak(DC = DC, DT = 0.001, N_T = 100, L_N = 9, a = 2.)
        for _ in range(1):
            emb.gen_data()        
            emb.run()
    
