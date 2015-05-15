import numpy as np
import particle_filter_new as PF

# For the particle filter module, this class mediates the emission probabilities
class PoissonLP(PF.LikelihoodPotential):
    """
    Poisson likelihood potential for use in the burak EM
    """
    def __init__(self, D_O, L0, L1, DT, G, spike_energy):
        """
        D_O - dimension of output (number of neurons)
        L0 - lower firing rate
        L1 - higher firing rate
        DT - time step
        G - gain factor
        spike_energy - function handle to spike energy function
                         see prob for desired arguments for spike_energy
        """
        PF.LikelihoodPotential.__init__(self, 2, D_O)
        self.L0 = L0
        self.L1 = L1
        self.DT = DT
        self.G = G
        self.spike_energy = spike_energy
        
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
        
        _R = np.zeros((self.D_O, 1)).astype('float32')
        _R[:, 0] = Yc
        # to pass to spike_prob function, N_P batches, 1 time step
        Es = - self.spike_energy(_XR, _YR, _R, self.L0, self.L1, self.DT, self.G)
        Es = Es - Es.mean()
        return np.exp(Es)
