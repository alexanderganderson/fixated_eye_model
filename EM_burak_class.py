# Python script containing a class that does expectation maximization
#   to estimate an image from simulated LGN cell responses
# See the end of the script for a sample usage

import numpy as np
import theano
import theano.tensor as T
from scipy.signal import convolve2d
from utils.bounded_diffusion import Center
import utils.particle_filter as PF
from utils.theano_gradient_routines import ada_delta
from utils.image_gen import ImageGenerator
from utils.SNR import SNR
import matplotlib.pyplot as plt
import cPickle as pkl
from scipy.io import savemat
from utils.time_filename import time_string

# For the particle filter module, this class mediates the emission probabilities
class PoissonLP(PF.LikelihoodPotential):
    """
    Poisson likelihood potential for use in the burak EM
    """
    def __init__(self, _L0, _L1, _DT, _G, _spike_energy):
        """
        L0 - lower firing rate
        L1 - higher firing rate
        DT - time step
        G - gain factor
        _spike_energy - function handle to spike energy function

        """
        PF.LikelihoodPotential.__init__(self)
        self.L0 = _L0
        self.L1 = _L1
        self.DT = _DT
        self.G = _G
        self.spike_energy = _spike_energy
        
    def prob(self, Yc, Xc):
        """
        Gives probability p(R_t|X_t, theta),
        theta = image
        Yc - Observed spiking pattern - R_t
        Vectorized according to samples (eg. each row of Y, X is a sample),
            outputs a vector of probabilities
        """
        self.N_P, _ = Xc.shape
        # Note to pass to theano function need:
        #  XR -> (N_batches, N_timesteps)
        #  R -> (N_b, N_pix, N_t)
        # here, we have one timestep, and particles correspond to batches,
        #  and given how R is, we need to broadcast the spikes over batches
        
        _XR = np.zeros((self.N_P, 1)).astype('float32')
        _YR = np.zeros((self.N_P, 1)).astype('float32')
        _XR[:, 0] = Xc[:, 0]
        _YR[:, 0] = Xc[:, 1]
        self.N_pix = Yc.shape[0]
        
        
        _R = np.zeros((self.N_pix, 1)).astype('float32')
        _R[:, 0] = Yc
        # to pass to spike_prob function, N_P batches, 1 time step
        Es = - self.spike_energy(_XR, _YR, _R, self.L0, self.L1, self.DT, self.G)
        Es = Es - Es.mean()
        return np.exp(Es)


class EMBurak:
    def __init__(self, _DT = 0.002, _DC = 40., _N_T = 200,
                 _L_I = 14, _L_N = 18):
        """
        Initializes the parts of the EM algorithm
            -- Sets all parameters
            -- Defines relevant theano variables
            -- Compiles theano functions
            -- Sets the gain factor for the circuit
            -- Initializes the Particle Filter Module
            -- Initializes the Image
        """

        self.debug = False # If True, show debug images
        self.save_mode = True # If true, save results of each EM iteration
        
        # Simulation Parameters
        self.DT = _DT # Simulation timestep
        self.DC = _DC  # Diffusion Constant
        self.L0 = 10.
        self.L1 = 100.
        self.ALPHA  = 0. # Image Regularization
        #self.BETA   = 100 # Pixel out of bounds cost param (pixels in 0,1)
        self.GAMMA = 100. # Pixel out of bounds cost parameter

        self.N_T = _N_T # Number of time steps
        self.L_I = _L_I # Linear dimension of image
        self.L_N = _L_N # Linear dimension of neuron receptive field grid

        self.N_B = 1 # Number of batches of data (must be 1)

        # EM Parameters
        # M - Parameters (ADADELTA)
        self.Rho = 0.4
        self.Eps = 0.001
        self.N_g_itr = 5
        self.N_itr = 20

        # E Parameters (Particle Filter)
        self.N_P = 25 # Number of particles for the EM
        
        # Other Parameters
        self.N_Pix = self.L_I ** 2 # Number of pixels in image
        self.N_N = self.L_N ** 2 # Number of neurons
        #self.N_L = self.N_Pix # Number of latent factors


        # Initialize pixel and LGN positions
        self.XS = np.arange(- self.L_I / 2, self.L_I / 2)
        self.YS = np.arange(- self.L_I / 2, self.L_I / 2)
        self.XS, self.YS = self.XS.astype('float32'), self.YS.astype('float32') # Position of pixels

        # Position of LGN receptive fields
        self.XE, self.YE = np.meshgrid(np.arange(- self.L_N / 2, self.L_N / 2),
                                       np.arange(- self.L_N / 2, self.L_N / 2))
        self.XE, self.YE = self.XE.ravel().astype('float32'), self.YE.ravel().astype('float32') 
        
        # Pixel values
        self.S = np.zeros((self.L_I, self.L_I)).astype('float32') 
        # Assumes that the first dimension is 'Y' and the second dimension is 'X'

        self.Var = 0.25 * np.ones((self.L_I,)).astype('float32') # Pixel spread variances
        self.G = 1. # Gain factor ... depends on Sig. makes inner products have max about 1... auto set later

        self.XR = np.zeros((self.N_B, 
                            self.N_T)).astype('float32') # X-Position of retina
        self.YR = np.zeros((self.N_B, 
                            self.N_T)).astype('float32') # Y-Position of retina
        self.R = np.zeros((self.N_N, self.N_T)).astype('float32')  # Spikes (1 or 0)
        self.Wbt = np.ones((self.N_B, self.N_T)).astype('float32') # Weighting for batches and time points
        #self.D = np.zeros((self.N_L, self.N_Pix)).astype('float32')  # Dictionary going from latent factors to image
        #self.A = np.zeros((self.N_L,)).astype('float32')      # Sparse Coefficients
        
        
        self.init_theano_vars()
        self.init_theano_funcs()
        self.set_gain_factor()
        self.init_particle_filter()
        
        self.init_image()
        
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
        
             
    def run(self):
        """
        Runs an iteration of EM
        Then saves relevant summary information about the run
        """
        self.run_EM()
        if (self.save_mode):
            self.save()


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

        self.t_S = theano.shared(self.S, 'S')

        self.t_XR = T.matrix('XR')
        self.t_YR = T.matrix('YR')
        self.t_R = T.matrix('R')
        #self.t_D = theano.shared(self.D, 'D1')
        #self.t_A = theano.shared(self.A, 'A')

        self.t_Wbt = T.matrix('Wbt')

        # Theano Parameters
        self.t_L0 = T.scalar('L0')
        self.t_L1 = T.scalar('L1')
        self.t_DT = T.scalar('DT')
        self.t_DC = T.scalar('DC')
        self.t_ALPHA = T.scalar('ALPHA')
        #self.t_BETA = T.scalar('BETA')
        self.t_GAMMA = T.scalar('GAMMA')
        self.t_G = T.scalar('G')

        # Take dot product of image with an array of gaussians
        
        # Note in this computation, we do the indices in this form:
        #  b, i, j, t
        #  batch, pixel, neuron, timestep
        self.t_dX = (self.t_XS.dimshuffle('x', 0, 'x', 'x') 
                     - self.t_XE.dimshuffle('x', 'x', 0, 'x') 
                     - self.t_XR.dimshuffle(0, 'x', 'x', 1))
        self.t_dX.name = 'dX'

        self.t_dY = (self.t_YS.dimshuffle('x', 0, 'x', 'x') 
                     - self.t_YE.dimshuffle('x', 'x', 0, 'x') 
                     - self.t_YR.dimshuffle(0, 'x', 'x', 1))
        self.t_dY.name = 'dY'

        # Use outer product trick to dot product image with point filters
        self.t_PixRFCouplingX = T.exp(-0.5 * self.t_dX ** 2 / 
                                       self.t_Var.dimshuffle('x', 0, 'x', 'x'))
        self.t_PixRFCouplingY = T.exp(-0.5 * self.t_dY ** 2 / 
                                       self.t_Var.dimshuffle('x', 0, 'x', 'x'))
        self.t_PixRFCouplingX.name = 'PixRFCouplingX'
        self.t_PixRFCouplingY.name = 'PixRFCouplingY'


        # Matrix of inner products between the images and the retinal RFs
        # indices: b, j, t
        self.t_IpsY = T.sum(self.t_S.dimshuffle('x', 0, 1, 'x', 'x') * 
                            self.t_PixRFCouplingY.dimshuffle(0, 1, 'x', 2, 3), axis = 1)
        #self.t_IpsX.name = 'IpsX'
        self.t_Ips = T.sum(self.t_IpsY * self.t_PixRFCouplingX, axis = 1)
        self.t_Ips.name = 'Ips'
        
        # Firing probabilities indexed by
        # b, j, t
        self.t_FP_0 = self.t_DT * T.exp(T.log(self.t_L0) + 
                                        T.log(self.t_L1 / self.t_L0) * 
                                        self.t_G * self.t_Ips)

        self.t_FP = T.switch(self.t_FP_0 > 0.9, 0.9, self.t_FP_0)

        # Compute Energy Functions (negative log-likelihood) to minimize
        # Note energy is weighted in time and batch by W_bt (used by particle filter)
        self.t_E_R = -T.sum(T.sum(self.t_R.dimshuffle('x', 0, 1) * T.log(self.t_FP) 
                               + (1 - self.t_R.dimshuffle('x', 0, 1)) * T.log(1 - self.t_FP), axis = 1) * self.t_Wbt)
        self.t_E_R.name = 'E_R'

        self.t_E_rec = (self.t_ALPHA * T.mean((self.t_S - 0.5) ** 2) + 
                        self.t_GAMMA * (T.sum(T.switch(self.t_S < 0., -self.t_S, 0)) + 
                                        T.sum(T.switch(self.t_S > 1., self.t_S - 1, 0)))
                        )
                            
                            
        self.t_E_rec.name = 'E_rec'

        self.t_E = self.t_E_rec + self.t_E_R
        self.t_E.name = 'E'

        # Auxiliary Theano Variables

        # Cost from poisson terms separated by batches for particle filter log probability
        self.t_E_R_b = -T.sum(self.t_R.dimshuffle('x', 0, 1) * T.log(self.t_FP) 
                              + (1 - self.t_R.dimshuffle('x', 0, 1)) * T.log(1 - self.t_FP), axis = (1, 2))

        # Generate Spikes
        self.rng = T.shared_randomstreams.RandomStreams(seed = 100)
        self.t_R_gen = (self.rng.uniform(size = self.t_FP.shape) < self.t_FP).astype('float32')

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
        self.costs = theano.function(inputs = [self.t_XR, self.t_YR,
                                               self.t_R, self.t_Wbt,
                                               self.t_L0, self.t_L1, 
                                               self.t_DT, self.t_G, 
                                               self.t_ALPHA, self.t_GAMMA],
                                     outputs = [self.t_E, self.t_E_rec, 
                                                self.t_E_R])


        # Returns the energy E_R = -log P(r|x,s) separated by batches
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

        self.img_grad = theano.function(inputs = [self.t_XR, self.t_YR,
                                             self.t_R, self.t_Wbt,
                                             self.t_L0, self.t_L1, 
                                             self.t_DT, self.t_G, 
                                             self.t_ALPHA, self.t_GAMMA,
                                             self.t_Rho, self.t_Eps],
                                             outputs = [self.t_E, self.t_E_rec, self.t_E_R],
                                             updates = self.s_grad_updates)

        
    

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
        """
        # Define necessary components for the particle filter
        D_H = 2 # Dimension of hidden state (i.e. x,y = 2 dims)
        sdev = np.sqrt(self.DC * self.DT / 2) # Needs to be sdev per component
        ipd = PF.GaussIPD(sdev * 0.001, 2)
        tpd = PF.GaussTPD(sdev, 2)
        ip = PF.GaussIP(sdev * 0.01, 2)
        tp = PF.GaussTP(sdev, 2)
        lp = PoissonLP(self.L0, self.L1, self.DT, self.G, self.spike_energy)
        self.pf = PF.ParticleFilter(ipd, tpd, ip, tp, lp)


    def init_image(self):
        """
        Initialize the Image
        """
        self.ig = ImageGenerator(self.L_I)
        self.ig.make_big_E()
        #self.ig.random()
        #self.ig.smooth()
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
            

    def gen_path(self):
        """
        Generate a retinal path. Note that the path has a bias towards the
            center so that the image does not go too far out of range
        """
        self.c = Center(self.L_I, self.DC, self.DT)
        for b in range(self.N_B):
            for t in range(self.N_T):
                x = self.c.get_center()
                self.XR[b, t] = x[0]
                self.YR[b, t] = x[1]
                self.c.advance()
            self.c.reset()


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
        self.R = self.spikes(self.XR, self.YR, self.L0, self.L1, self.DT, self.G)[0]
        print 'Mean firing rate ' + str(self.R.mean() / self.DT)


    def true_costs(self):
        """
        Prints out the negative log-likelihood of the observed spikes given the
            image and path that generated them
        Note that the energy is normalized by the number of timesteps
        """
        print 'Pre-EM testing'
        self.t_S.set_value(self.S)
        E, E_rec, E_R = self.costs(self.XR, self.YR, 
                                   self.R, self.Wbt, 
                                   self.L0, self.L1, 
                                   self.DT, self.G, 
                                   self.ALPHA, self.GAMMA)
        # FIXME: what is the meaning of E here?         
        print 'Costs of underlying data ' + str((E/self.N_T, E_rec/self.N_T))


    def reset_image_estimate(self):
        """
        Resets the value of the image as stored on the GPU
        """
        self.t_S.set_value(0.5 + np.zeros(self.S.shape).astype('float32'))

   
    def run_E(self, t):
        """
        Runs the expectation step for the first t time steps
        t - number of timesteps
        The result is saved in self.pf.XS,WS,means
        """
        if (t > self.N_T):
            print 'Maximum simulated timesteps exceeded in E step'
        self.pf.run(self.R.transpose()[0:t], self.N_P)
        print 'Path SNR ' + str(SNR(self.XR[0][0:t], self.pf.means[0:t, 0]))
    
    
    def reset_M_aux(self):
        """
        Resets auxillary gradient descent variables for the M step
            eg. for ADADelta, we reset the RMS of dx and g
        """
        self.t_S_Eg2.set_value(np.zeros_like(self.S).astype('float32'))
        self.t_S_EdS2.set_value(np.zeros_like(self.S).astype('float32'))
        
    
    def run_M(self, t, N_g_itr = 5):
        """
        Runs the maximization step for the first t time steps
        resets the values of auxillary gradient descent variables at start
        t - number of time steps
        result is saved in t_S.get_value()
        """
        self.reset_M_aux()
        print 'Spike Energy / t | Reg. Energy / t | SNR' 
        for v in range(N_g_itr):
            E, E_rec, E_R = self.img_grad(
                       self.pf.XS[:, :, 0].transpose()[:, 0:t],
                       self.pf.XS[:, :, 1].transpose()[:, 0:t],
                       self.R[:, 0:t], self.pf.WS.transpose()[:, 0:t],
                       self.L0, self.L1, self.DT, 
                       self.G, 0.5 * self.ALPHA * t / self.N_T, self.GAMMA,
                       self.Rho, self.Eps)
            self.img_SNR = SNR(self.S, self.t_S.get_value())
            print (str(E_R / t) + ' ' + 
                   str(E_rec / t) + ' ' +  
                   str(self.img_SNR))

        
    def run_EM(self, N_itr = 20, N_g_itr = 5):
        """
        Runs full expectation maximization algorithm
        N_itr - number of iterations of EM
        N_g_itr - number of gradient steps in M step
        Saves summary of run info in self.data 
        Note running twice will overwrite this run info
        """
        
        self.N_itr = N_itr
        self.N_g_itr = N_g_itr
        
        self.reset_image_estimate()
            
        EM_data = {}
    
        print 'Running full EM'
        for u in range(N_itr):
            t = self.N_T * (u + 1) / N_itr #t = self.N_T
            print 'Iteration number ' + str(u) + ' t_step annealing ' + str(t)
            
            # Run E step
            self.run_E(t)
            
            # Run M step
            self.run_M(t, N_g_itr = N_g_itr)
            
            iteration_data = {}
            iteration_data['time_steps'] = t
            iteration_data['path_means'] = self.pf.means
            iteration_data['path_sdevs'] = self.pf.sdevs
            iteration_data['image_est'] = self.t_S.get_value()

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
        data['ALPHA'] = self.ALPHA
        data['GAMMA'] = self.GAMMA
        
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
        fn = self.output_dir + '/data.' + time_string() + '.pkl'        
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
        plt.imshow(np.abs(self.t_S.get_value() - self.S).reshape(self.L_I, self.L_I), 
                   cmap = plt.cm.gray, interpolation = 'nearest')
        plt.colorbar()
        plt.show()



    def true_path_infer_image_costs(self, N_g_itr = 10):
        """
        Infers the image given the true path
        Prints the costs associated with this step
        """
        self.reset_img_gpu()
        print 'Original Path, infer image'
        t = self.N_T
        self.run_M(t)


    def true_image_infer_path_costs(self):
        print 'Original image, Infer Path'
        print 'Path SNR'
        self.t_S.set_value(self.S)
        for _ in range(4):
            self.run_E(self.N_T)

        if self.debug:
            self.pf.plot(self.XR[0], self.YR[0], self.DT)
    
   
    def calculate_inner_products(self):
        """
        Calculates the inner products used
        """
        self.Ips, self.FP = self.RFS(self.XR, self.YR, 
                                     self.L0, self.L1, 
                                     self.DT, self.G)
 

if __name__ == '__main__':
    emb = EMBurak(_DC = 50., _DT = 0.001, _N_T = 100)
    for _ in range(2):
        emb.gen_data()
        emb.run()
    