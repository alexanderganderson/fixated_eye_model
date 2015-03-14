import numpy as np
import theano
import theano.tensor as T
from scipy.signal import convolve2d
from utils.bounded_diffusion import Center
import utils.particle_filter as PF
from utils.theano_gradient_routines import ada_delta
from utils.image_gen import ImageGenerator

debug = False # If True, show debug images

#if debug:
import matplotlib.pyplot as plt

# Simulation Parameters
DT = 0.005 # Simulation timestep
DC = 20.  # Diffusion Constant
L0 = 10.
L1 = 100.
ALPHA  = 100 # Image Regularization
BETA   = 100 # Pixel out of bounds cost param (pixels in 0,1)

N_T = 300 # Number of time steps
L_I = 14 # Linear dimension of image
L_N = 18 # Linear dimension of neuron receptive field grid

N_B = 1 # Number of batches of data (must be 1)

# EM Parameters
# M - Parameters (ADADELTA)
Rho = 0.9
Eps = 0.01
N_g_itr = 5
N_itr = 20
# E Parameters (Particle Filter)
N_P = 25 # Number of particles for the EM
N_Pix = L_I ** 2 # Number of pixels in image
N_N = L_N ** 2 # Number of neurons
N_L = N_Pix # Number of latent factors


# Initialize pixel and LGN positions
#XS, YS = np.meshgrid(np.arange(- L_I / 2, L_I / 2),
#                     np.arange(- L_I / 2, L_I / 2))
XS = np.arange(- L_I / 2, L_I / 2)
YS = np.arange(- L_I / 2, L_I / 2)
XS, YS = XS.ravel().astype('float32'), YS.ravel().astype('float32') # Position of pixels
XE, YE = np.meshgrid(np.arange(- L_N / 2, L_N / 2),
                     np.arange(- L_N / 2, L_N / 2))
XE, YE = XE.ravel().astype('float32'), YE.ravel().astype('float32') # Position of LGN receptive fields
#XE = 0.5 * XE
#YE = 0.5 * YE
S = np.zeros((L_I, L_I)).astype('float32') # Pixel values
# Assumes that the first dimension is 'X' and the second dimension is 'Y'

Var = 0.09 * np.ones((L_I,)).astype('float32') # Pixel spread variances
G = 1. # Gain factor ... depends on Sig. makes inner products have max about 1... auto set later

XR = np.zeros((N_B, N_T)).astype('float32') # X-Position of retina
YR = np.zeros((N_B, N_T)).astype('float32') # Y-Position of retina
R = np.zeros((N_N, N_T)).astype('float32')  # Spikes (1 or 0)
Wbt = np.ones((N_B, N_T)).astype('float32') # Weighting for batches and time points
D = np.zeros((N_L, N_Pix)).astype('float32')  # Dictionary going from latent factors to image
#for i in range(min(N_L, N_P)):
#    D[i, i] = 1.0
A = np.zeros((N_L,)).astype('float32')      # Sparse Coefficients


# Define Theano Variables
t_XS = theano.shared(XS, 'XS')
t_YS = theano.shared(YS, 'YS')
t_XE = theano.shared(XE, 'XE')
t_YE = theano.shared(YE, 'YE')
t_Var = theano.shared(Var, 'Sig')

t_S = theano.shared(S, 'S')

t_XR = T.matrix('XR')
t_YR = T.matrix('YR')
t_R = T.matrix('R')
t_D = theano.shared(D, 'D1')
t_A = theano.shared(A, 'A')

t_Wbt = T.matrix('Wbt')

# Theano Parameters
t_L0 = T.scalar('L0') #FIXME
t_L1 = T.scalar('L1')
t_DT = T.scalar('DT')
t_DC = T.scalar('DC')
t_ALPHA = T.scalar('ALPHA')
t_BETA = T.scalar('BETA')
t_G = T.scalar('G')


# Note in this computation, we do the indices in this form:
#  b, i, j, t
#  batch, pixel, neuron, timestep
t_dX = (t_XS.dimshuffle('x', 0, 'x', 'x') 
        - t_XE.dimshuffle('x', 'x', 0, 'x') 
        - t_XR.dimshuffle(0, 'x', 'x', 1))
t_dX.name = 'dX'

t_dY = (t_YS.dimshuffle('x', 0, 'x', 'x') 
        - t_YE.dimshuffle('x', 'x', 0, 'x') 
        - t_YR.dimshuffle(0, 'x', 'x', 1))
t_dY.name = 'dY'

t_PixRFCouplingX = T.exp(-0.5 * t_dX ** 2 / 
                         t_Var.dimshuffle('x', 0, 'x', 'x'))
t_PixRFCouplingY = T.exp(-0.5 * t_dY ** 2 / 
                         t_Var.dimshuffle('x', 0, 'x', 'x'))
t_PixRFCouplingX.name = 'PixRFCouplingX'
t_PixRFCouplingY.name = 'PixRFCouplingY'


# Matrix of inner products between the images and the retinal RFs
# indices: b, j, t
t_IpsX = T.sum(t_S.dimshuffle('x', 0, 1, 'x', 'x') * 
               t_PixRFCouplingX.dimshuffle(0, 1, 'x', 2, 3), axis = 1)
t_IpsX.name = 'IpsX'
t_Ips = T.sum(t_IpsX * t_PixRFCouplingY, axis = 1)
t_Ips.name = 'Ips'
# Firing probabilities indexed by
# b, j, t
t_FP_0 = t_DT * T.exp(T.log(t_L0) + T.log(t_L1 / t_L0) * t_G * t_Ips)

t_FP = T.switch(t_FP_0 > 0.9, 0.9, t_FP_0)

# Cost from poisson terms separated by batches 
t_E_R_b = -T.sum(t_R.dimshuffle('x', 0, 1) * T.log(t_FP) + (1 - t_R.dimshuffle('x', 0, 1)) * T.log(1 - t_FP), axis = (1, 2))

t_E_R = -T.sum(T.sum(t_R.dimshuffle('x', 0, 1) * T.log(t_FP) 
                       + (1 - t_R.dimshuffle('x', 0, 1)) * T.log(1 - t_FP), axis = 1) * t_Wbt)
t_E_R.name = 'E_R'


# In[9]:

rng = T.shared_randomstreams.RandomStreams(seed = 100)
t_R_gen = (rng.uniform(size = t_FP.shape) < t_FP).astype('float32')


# In[10]:

# Define terms in the cost function
#t_E_DX  = T.sum((t_XR[:, 1:] - t_XR[:, :-1]) ** 2) / (2 * t_DC * t_DT)
#t_E_DY  = T.sum((t_YR[:, 1:] - t_YR[:, :-1]) ** 2) / (2 * t_DC * t_DT)
#t_E_rec = t_ALPHA * T.mean((t_S - T.dot(t_A, t_D)) ** 2)
t_E_rec = t_ALPHA * T.mean(t_S ** 2)
t_E_p = t_BETA * (T.sum(T.switch(t_S < 0., -t_S, 0)) + 
                  T.sum(T.switch(t_S > 1., t_S - 1, 0))) # Image prior

t_E_rec.name = 'E_rec'


t_E_sp  = 0 * t_BETA * T.sum(T.abs_(t_A))
#t_E_DXY0 = (T.sum(t_XR[:, 0] ** 2) + T.sum(t_YR[:, 0] ** 2))/(10 * t_DC * t_DT) # Approx delta function as this
#t_E = t_E_DXY0 + 1. * (t_E_DX + t_E_DY) + t_E_rec + t_E_sp + t_E_R
t_E = t_E_rec + t_E_sp + t_E_R + t_E_p
t_E.name = 'E'


# In[11]:

RFS = theano.function(inputs = [t_XR, t_YR,
                                t_L0, t_L1, t_DT, t_G],
                        outputs = [t_Ips, t_FP])


# In[12]:

# Returns the energy E_R = -log P(r|x,s) separated by batches
# Function for the particle filter
spike_energy = theano.function(inputs = [t_XR, t_YR, t_R,
                                         t_L0, t_L1, t_DT, t_G],
                             outputs = t_E_R_b)


# In[13]:

# Initially generate spikes
gen_spikes = theano.function(inputs = [t_XR, t_YR,
                                       t_L0, t_L1, t_DT, t_G],
                             outputs = t_R_gen)


# In[14]:

costs = theano.function(inputs = [t_XR, t_YR,
                                  t_R, t_Wbt,
                                  t_L0, t_L1, t_DT, t_G, t_ALPHA, t_BETA],
                        outputs = [t_E, t_E_rec, t_E_p, t_E_R])


# In[15]:

t_Rho = T.scalar('Rho')
t_Eps = T.scalar('Eps')
t_ada_params = (t_Rho, t_Eps)


# In[16]:

s_grad_updates = ada_delta(t_E, t_S, *t_ada_params)
t_S_Eg2, t_S_EdS2, _ = s_grad_updates.keys()


# In[17]:

img_grad = theano.function(inputs = [t_XR, t_YR,
                                     t_R, t_Wbt,
                                     t_L0, t_L1, t_DT, t_G, t_ALPHA, t_BETA,
                                     t_Rho, t_Eps],
                           outputs = [t_E, t_E_rec, t_E_p, t_E_R],
                           updates = s_grad_updates)


# In[18]:

def SNR(S, S0):
    return np.var(S) / np.var(S - S0)


# In[19]:

# For the particle filter module, we create this class for the emission probabilities
class PoissonLP(PF.LikelihoodPotential):
    """
    Poisson likelihood potential for use in the burak EM
    """
    def __init__(self, _L0, _L1, _DT, _G):
        """
        s - image to help define the function?
        """
        PF.LikelihoodPotential.__init__(self)
        self.L0 = _L0
        self.L1 = _L1
        self.DT = _DT
        self.G = _G
        
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
        Es = -spike_energy(_XR, _YR, _R, self.L0, self.L1, self.DT, self.G)
        Es = Es - Es.mean()
        return np.exp(Es)
        


# In[20]:

# Initialize Image
ig = ImageGenerator(L_I)
ig.make_T()
#ig.random()
ig.smooth()
ig.normalize()
S = ig.img
t_S.set_value(S)
if (debug):
    ig.plot()


# In[21]:

# Generate X's
c = Center(L_I, DC, DT)
for b in range(N_B):
    for t in range(N_T):
        x = c.get_center()
        XR[b, t] = x[0]
        YR[b, t] = x[1]
        c.advance()
    c.reset()


# In[22]:

if (debug):
    plt.subplot(1,2,1)
    plt.plot(np.arange(N_T) * DT, XR[0])
    plt.title('x coordinate')
    plt.subplot(1,2,2)
    plt.plot(np.arange(N_T) * DT, YR[0])
    plt.title('y coordinate')
    plt.show()

# In[23]:

G = 1.
Ips, FP = RFS(XR, YR, L0, L1, DT, G)
G = (1 / Ips.max()).astype('float32')


# In[24]:

if debug:
    plt.title('Firing Probability Histogram')
    plt.hist(FP.ravel())
    plt.show()

# In[25]:

if debug:
    q = 0
    t = 1 * (N_T-1)
    plt.subplot(1, 2, 1)
    plt.title('IPs at T = ' + str(t))
    plt.imshow( (1/DT) * FP[q, :, t].reshape(L_N, L_N), cmap = plt.cm.gray, interpolation = 'nearest')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.title('Original Image')
    plt.imshow(S.reshape(L_I, L_I), cmap = plt.cm.gray, interpolation = 'nearest')
    plt.colorbar()
    plt.show()

# In[26]:

R = gen_spikes(XR, YR, L0, L1, DT, G)[0]
print 'Mean firing rate ' + str(R.mean()/DT)


# In[27]:

# Define necessary components for the particle filter
D_H = 2 # Dimension of hidden state (i.e. x,y = 2 dims)
sdev = np.sqrt(DC * DT)
ipd = PF.GaussIPD(sdev * 0.01, 2)
tpd = PF.GaussTPD(sdev, 2)
ip = PF.GaussIP(sdev * 0.01, 2)
tp = PF.GaussTP(sdev, 2)
lp = PoissonLP(L0, L1, DT, G)
pf = PF.ParticleFilter(ipd, tpd, ip, tp, lp)


# In[28]:

# Rearrange indices for use in the particle filter
R_ = np.transpose(R)


# In[29]:

print 'Pre-EM testing'
t_S.set_value(S)
E, E_rec, E_p, E_R = costs(XR, YR, R, Wbt, L0, L1, DT, G, ALPHA, BETA)
print 'Costs of underlying data ' + str((E/N_T, E_rec/N_T, E_p / N_T))


# In[30]:

t_S.set_value(0.5 + np.zeros(S.shape).astype('float32'))
t_S_Eg2.set_value(np.zeros(S.shape).astype('float32'))
t_S_EdS2.set_value(np.zeros(S.shape).astype('float32'))


# In[31]:

print 'Original Path, infer image'
print 'Spike Energy | Reg. Energy | Prior | SNR' 
t = N_T
for v in range(10):   
    E, E_rec, E_p, E_R = img_grad(XR[:, 0:t], YR[:, 0:t],
                                    R[:, 0:t], Wbt[:, 0:t],
                                    L0, L1, DT, G, ALPHA, BETA,
                                    Rho, Eps)
    print (str(E_R / N_T) + ' ' + 
           str(E_rec / N_T) + ' ' + 
           str(E_p / N_T) + ' ' + 
           str(SNR(S, t_S.get_value())))


# In[32]:

if debug:
    vmin = -1.
    vmax = 1.
    plt.subplot(1, 3, 1)
    plt.title('EM Estimate')
    plt.imshow(t_S.get_value().reshape(L_I, L_I), 
               cmap = plt.cm.gray, 
               interpolation = 'nearest', 
               vmin = vmin, vmax = vmax)
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.title('Actual')
    plt.imshow(S.reshape(L_I, L_I), cmap = plt.cm.gray, 
               interpolation = 'nearest',
               vmin = vmin, vmax = vmax)
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.title('Error')
    plt.imshow(np.abs(t_S.get_value() - S).reshape(L_I, L_I), cmap = plt.cm.gray, interpolation = 'nearest')
    plt.colorbar()
    plt.show()

# In[41]:

print 'Original image, Infer Path'
print 'Path SNR'
t_S.set_value(S)
for _ in range(4):
    pf.run(R_, N_P)
    print str(SNR(XR[0], pf.means[:, 0]))

if debug:
    pf.plot(XR[0], 0, DT)



t_S.set_value(0.5 + np.zeros(S.shape).astype('float32'))

print 'Full EM'


for u in range(N_itr):
#    t = N_T
    t = N_T * (u + 1) / N_itr
    print 'Iteration number ' + str(u) + ' t_step annealing ' + str(t)
    pf.run(R_[0:t], N_P)

    print 'Path SNR ' + str(SNR(XR[0][0:t], pf.means[0:t, 0]))
    
    t_S_Eg2.set_value(np.zeros_like(S).astype('float32'))
    t_S_EdS2.set_value(np.zeros_like(S).astype('float32'))



    for v in range(N_g_itr):   
        E, E_rec, E_sp, E_R = img_grad(pf.XS[:, :, 0].transpose()[:, 0:t],
                                       pf.XS[:, :, 1].transpose()[:, 0:t],
                                       R[:, 0:t], pf.WS.transpose()[:, 0:t],
                                       L0, L1, DT, G, 0.5 * ALPHA * t / N_T, BETA * t / N_T,
                                       Rho, Eps)
        print 'Spike Energy per timestep ' + str(E_R / t) + ' Img Reg per timestep ' + str(E_rec / t)
    
    print 'Image SNR ' + str(SNR(S, t_S.get_value()))

if debug:
    pf.run(R_, 25)
    pf.plot(XR[0], 0, DT)
    plt.show()

if debug:
    vmin = -0.1
    vmax = 0.1
    plt.subplot(1, 3, 1)
    plt.title('EM Estimate')
    plt.imshow(t_S.get_value().reshape(L_I, L_I), 
               cmap = plt.cm.gray, 
               interpolation = 'nearest', 
               vmin = vmin, vmax = vmax)
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.title('Actual')
    plt.imshow(S.reshape(L_I, L_I), cmap = plt.cm.gray, 
               interpolation = 'nearest',
               vmin = vmin, vmax = vmax)
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.title('Error')
    plt.imshow(np.abs(t_S.get_value() - S).reshape(L_I, L_I), cmap = plt.cm.gray, interpolation = 'nearest')
    plt.colorbar()
    plt.show()
#    plt.savefig('img_est.png')

if debug:
    plt.plot(np.sum(pf.WS ** 2, axis = 1) ** -1)
    plt.show()

