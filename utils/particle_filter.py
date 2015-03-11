# Code for a particle filter
#
# A particle filter lets us sample the distribution of hidden states
#    given the observed state in a Markov chain where the hidden 
#    state is continuous valued.
# The file defines class prototypes to be implemented to run the filter

import numpy as np
import matplotlib.pyplot as plt


def gauss(X, s):
    """
    Gaussian PDF with sdev = s
    """
    return np.exp(- 0.5 * X ** 2 / s ** 2) / np.sqrt(2 * np.pi * s ** 2)


class PropDist:
    """
    Class prototype for proposal distributions
    """
    def __init__(self, _dim):
        """
        Create a proposal distribution object
        _dim = dimension of hidden state of the Markov model
        """
        self.dim = _dim
    
    def prob(self):
        return 0
    
    def sample(self, shape):
        return np.zeros(size = dim)


class InitPropDist(PropDist):
    def __init__(self, _dim):
        """
        Create an object for the proposal distribution for the first
            hidden state. 
        """
        PropDist.__init__(self, _dim)
    
    def prob(self, X0, Y0):
        """
        Return proposal probability q(X0|Y0)
        X0 - Hidden Initial State
        Y0 - Initial Observation
        """
        return 0
    
    def sample(self, Y0, N_S):
        """
        Sample from proposal distribution given Y0
        Y0 - Initial observation
        N_S - number of samples to return
        Returns samples as rows
        """
        return np.zeros((N_S, self.dim))


class GaussIPD(InitPropDist):
    def __init__(self, _sigma, _dim):
        """
        Create a simple Gaussian proposal distribution for the
            initial state:
        q(X_0|Y_0) = N(0, sigma)
        _sigma - variance
        _dim - dimension of hidden states
        """
        InitPropDist.__init__(self, _dim)
        self.sigma = _sigma
    
    def prob(self, X0, Y0):
        """
        Return the proposal probability q(X0|Y0)
        X0 - Initial hidden state
        Y0 - Initial observed state
        """
        return np.prod(gauss(X0, self.sigma), axis = 1)
    
    def sample(self, Y0, N_S):
        """
        Sample from proposal distribution given Y0
        Y0 - Initial observation
        N_S - number of samples to return
        Returns samples as rows
        """
        return self.sigma * np.random.randn(N_S, self.dim)


class TransPropDist(PropDist):
    def __init__(self, _dim):
        """
        Object for the proposal distribution for transitions between
            hidden states:
        q(Xc|Yc, Xp), 
        Xc - current hidden state
        Yc - current observed state
        Xp - previous hidden state
        """
        PropDist.__init__(self, _dim)
    
    def prob(self, Xc, Yc, Xp):
        """
        Return value of q(Xc|Yc, Xp)
        """
        return 0
    
    def sample(self, Yc, Xp):
        """
        Return samples Xc ~ q(Xc|Yc, Xp)
        Note Xp has rows that are samples and columns that are the
            dimensions of the hidden state
        """
        return np.zeros(Xp.shape)


class GaussTPD(TransPropDist):
    def __init__(self, _sigma, _dim):
        """
        Simple gaussian transition proposal distribution:
            q(Xc|Yc, Xp) ~ N(mu = Xp, sigma)
        _sigma - standard deviation for the proposal
        _dim - dimension of hidden state of the markov model
        """
        TransPropDist.__init__(self, _dim)
        self.sigma = _sigma
        
    def prob(self, Xc, Yc, Xp):
        """
        Returns the probability N(mu = Xp, sigma)(Xc)
        """
        return np.prod(gauss(Xc - Xp, self.sigma), axis = 1)
    
    def sample(self, Yc, Xp):
        """
        Samples from the proposal distribution
        Yc - Current observed state
        Xp - previous hidden state, each particle comes in as a row
        """
        return np.random.randn(*Xp.shape) * self.sigma + Xp


# In addition to proposal distributions, we have potentials that give us
#   the probabilities of the HMM transition and the emission probabilities

class Potential:
    """
    Simple Class prototype for potentials
    """
    def __init__(self):
        0
        
    def prob(self):
        return 0


class InitialPotential(Potential):
    """
    Class prototype for the prior on the initial hidden state of the HMM,
    p(X0)
    """
    def __init__(self, _dim):
        Potential.__init__(self)
        self.dim = _dim
        
    def prob(self, Xc):
        return 0


class GaussIP(InitialPotential):
    def __init__(self, _sigma, _dim):
        """
        Simple Prior for the initial hidden state
            p(X0) = N(0, _sigma)
        _sigma - standard deviation
        _dim - dimension of hidden state
        """
        self.sigma = _sigma
        InitialPotential.__init__(self, _dim)
    
    def prob(self, Xc):
        return np.prod(gauss(Xc, self.sigma), axis = 1)

class TransPotential(Potential):
    """
    Class prototype for the transition probability potential:
    p(Xc|Xp), Xc - current state, Xp - previous state
    """
    def __init__(self, _dim):
        Potential.__init__(self)
        self.dim = _dim
        
    def prob(self, Xc, Xp):
        """
        Return probability p(Xc|Xp)
        """
        return 0


class GaussTP(TransPotential):
    """
    Gaussian transition probability potential:
    p(Xc|Xp) = N(mu = Xp, sigma)
    """
    def __init__(self, _sigma, _dim):
        """
        _sigma - standard deviation
        _dim - dimension of hidden state
        """
        TransPotential.__init__(self, _dim)
        self.sigma = _sigma
        
    def prob(self, Xc, Xp):
        """
        Return p(Xc|Xp)
        """
        return np.prod(gauss(Xc - Xp, self.sigma), axis = 1)


class LikelihoodPotential(Potential):
    """
    Class prototype for likelihood function p(Yc|Xc)
    Yc - current observed state
    Xc - current hidden state
    """
    def __init__(self):
        Potential.__init__(self)
    
    def prob(self, Yc, Xc):
        """
        Return p(Yc|Xc)
        """
        return 0
    
class GaussLP(LikelihoodPotential):
    """
    Simple gaussian likelihood p(Yc|Xc) = N(mu = Xc, _sigma)
    """
    def __init__(self, _sigma):
        LikelihoodPotential.__init__(self)
        self.sigma = _sigma
    
    def prob(self, Yc, Xc):
        return np.prod(gauss(Yc - Xc, self.sigma), axis = 1)


class ParticleFilter:
    """
    Actual class for the particle filter
    Stores the potentials and proposal distributions to run the HMM
    N_T - number of time steps
    N_P - number of particles
    dim - number of dimensions of hidden state
    After calling run with some output data, the class stores the 
        weights and the associated particles in the arrays
    self.XS - weights in form (N_T, N_P, dim)
    self.WS - weights in the form (N_T, N_P)
    Note XS[i], WS[i] correspond to samples and weights from the
        following distribution:
        p(X_i|Y_0,Y_1,...,Y_i)
    """
    def __init__(self, _ipd, _tpd, _ip, _tp, _lp):
        """
        Create a Particle Filter Object
        ipd - Initial Proposal Distribution
        tpd - Transition Proposal Distribution
        ip - Initial Potential
        tp - Transition Potential
        lp - Likelihood Potential
        """
        self.ipd = _ipd
        self.tpd = _tpd
        self.ip = _ip
        self.tp = _tp
        self.lp = _lp
        self.dim = self.ipd.dim
        
    def resample(self, W):
        """
        Implements the systematic resampling method
        Given a normalized weight matrix W (shape (N_S)), return 
            indices corresponding to a resampling
        Eg. if we have our samples X, then
        X[idx] gives us a resampling of those samples
        """
        u = np.random.random() / self.N_P
        U = u + np.arange(self.N_P) / (1. * self.N_P)
        S = np.cumsum(W)

        i = 0
        j = 0
        idx = np.zeros(self.N_P, dtype = 'int')

        while(j < self.N_P and i < self.N_P):
            if (U[j] <= S[i]):
                idx[j] = i
                j = j + 1
            else:
                i = i + 1
        return idx
    
    def run(self, Y, _N_P):
        """
        Y - matrix of observed values of the form
        N_P - number of particles to use
        Y_T = np.zeros(N_T, other dims)
        
        Performs a resampling if the effective sample size becomes
            less than have the number of particles
        """
        self.resample_times = []
        self.N_P = _N_P
        self.N_T = Y.shape[0]
        self.XS = np.zeros((self.N_T, self.N_P, self.dim)).astype('float32')
        self.WS = np.zeros((self.N_T, self.N_P)).astype('float32')
        
        self.XS[0] = self.ipd.sample(Y[0], self.N_P)
        self.WS[0] = (self.ip.prob(self.XS[0]) 
                      * self.lp.prob(Y[0], self.XS[0]) 
                      / self.ipd.prob(self.XS[0], Y[0])
                      )
        self.WS[0] = self.WS[0] / np.sum(self.WS[0])
        
        for i in range(1, self.N_T):
            self.XS[i] = self.tpd.sample(Y[i], self.XS[i - 1])
            self.WS[i] = (self.WS[i - 1] 
                          * self.lp.prob(Y[i], self.XS[i]) 
                          * self.tp.prob(self.XS[i], self.XS[i - 1]) 
                          / self.tpd.prob(self.XS[i], Y[i], self.XS[i - 1])
                          )
            self.WS[i] = self.WS[i] / np.sum(self.WS[i])
            if (np.sum(self.WS[i] ** 2) ** (-1) < self.N_P / 2.):
                idx = self.resample(self.WS[i])
                self.XS[i] = self.XS[i, idx]
                self.WS[i] = 1. / self.N_P
                self.resample_times.append(i)
    
        self.means = np.sum(self.XS * 
                            self.WS.reshape(self.N_T, self.N_P, 1), 
                            axis = 1)
        self.sdevs = np.sqrt(np.sum(self.XS ** 2 * 
                            self.WS.reshape(self.N_T, self.N_P, 1), 
                                 axis = 1) - self.means ** 2)
    
            
    def plot(self, X, d, DT):
        """
        Generate a plot of the estimated hidden state and the real hidden state
            as a function of time
        X - actual hidden state along the dimension that we want
        #Y - observed state
        d - index for which dimension of hidden state to plot
        DT - timestep size
        """
        
                
        
        plt.fill_between(DT * np.arange(self.N_T), 
                         self.means[:, 0] - self.sdevs[:, 0], 
                         self.means[:, 0] + self.sdevs[:, 0], 
                         alpha = 0.5, linewidth = 1.)
        plt.plot(DT * np.arange(self.N_T), self.means[:, 0], label = 'estimate')
        plt.plot(DT * np.arange(self.N_T), X, label = 'actual')
        plt.xlabel('Time (s)')
        plt.ylabel('Relative position (pixels)')
        plt.legend()
        plt.title('Estimated versus actual position given the image')
        #plt.show()

def main():
    # A quick example: consider a HMM with gaussian transitions, and 
    #   gaussian noise added onto the outputs
    s1 = 0.01 # Hidden state transition standard deviation
    s2 = 0.03 # Noise for observed state
    N_T = 100 # Number of time steps
    N_P = 50 # Number of particles
    D_H = 2 # Dimension of hidden state


    # Generate some data according to this model
    X = np.zeros((N_T, D_H))
    Y = np.zeros((N_T, D_H))
    for i in range(D_H):
        X[:, i] = np.cumsum(s1 * np.random.randn(N_T))
    Y = np.random.randn(N_T, D_H) * s2 + X

    # Create the appropriate proposal distributions and potentials
    ipd = GaussIPD(s1, D_H)
    tpd = GaussTPD(s1, D_H)
    ip = GaussIP(s1, D_H)
    tp = GaussTP(s1, D_H)
    lp = GaussLP(s2)


    pf = ParticleFilter(ipd, tpd, ip, tp, lp)
    pf.run(Y, N_P)
    pf.plot(X, Y, 0)


if(__name__ == '__main__'):
    main()
