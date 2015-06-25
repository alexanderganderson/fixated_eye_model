import numpy as np
import matplotlib.pyplot as plt
import cPickle as pkl
import abc

class Center:
    def __init__(self, Lx, D, dt):
        """
        Class that Implements a diffusing center in a box of size Lx
        
        Lx = linear dimension of square to diffuse in
        D = diffusion constant
        dt = timestep size
        x = coordinates of the current location of the random walk, 
            initialized as [0, 0]
        Initializes Center Object
        """
        self.Lx = Lx
        self.D = D
        self.m0 = np.array([0, 0], dtype = 'float64') # current position
        self.dt = dt
        # The diffusion is biased towards the center by taking a product of gaussians
        #   A product of gaussians is also a gaussian with mean, sdev given as (mn, sn)
        self.m1 = np.array([0, 0], dtype = 'float64') # center of image
        self.s0 = np.sqrt(self.D * self.dt) # Standard deviation for diffusion
        self.s1 = self.Lx / 4  # Standard Deviation for centering gaussian
        self.sn = 1 / np.sqrt(1 / self.s0 ** 2 + 1 / self.s1 ** 2)

        
    def advance(self):
        """
        Updates location according to a random walk that stays within a box
        """
        
        self.mn = (self.m0 / self.s0 ** 2 + self.m1 / self.s1 ** 2) * self.sn ** 2
        while(True):
            # Note that for 2d diffusion, each component's variance is half the
            #   variance of the overall step length
            temp = self.mn + np.random.normal(size = 2, scale = self.sn / np.sqrt(2))
            if (temp[0] > - self.Lx / 2 
                and temp[0] < self.Lx / 2 
                and temp[1] > - self.Lx / 2 
                and temp[1] < self.Lx / 2):
                self.m0 = temp
                break

    def get_center(self):
        return self.m0
    
    def reset(self):
        self.m0 = np.array([0, 0], dtype = 'float64')
    
    def __str__(self):
        return str(self.x)

class PathGenerator():
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, N_T, *args):
        """
        Initialize a 2d Path Generator
        """
        self.N_T = N_T

    @abc.abstractmethod
    def gen_path(self):
        """
        Generates a 2d path with N_T timesteps starting at (0,0)
        """
        return np.zeros((2, self.N_T))

    @abc.abstractmethod:
    def path_mode(self):
        """
        Returns a string describing the path
        """
        return ' '

class DiffusionPathGenerator(PathGenerator):
    def __init__(self, N_T, Lx, D, dt):
        """
        Creates a path generator that does bounded diffusion
        """
        PathGenerator.__init__(self, N_T)
        self.c = Center(Lx, D, dt)
        
    def gen_path(self):
        self.c.reset()
        path = np.zeros((2, self.N_T))
        for i in range(self.N_T):
            path[:, i] = self.c.get_center()
            self.c.advance()
        return path
    
    def path_mode(self):
        return 'Diffusion'

class ExperimentalPathGenerator(PathGenerator):
    def __init__(self, N_T, filename, dt):
        """
        Creates a path generator that uses real experimental data
        filename - filename for pkl file that contains an array of paths
              data['paths'] = (N_runs, 2, number of timesteps)
        """
        PathGenerator.__init__(self, N_T)
        self.data = pkl.load(filename)
        self.dt = data['dt']
        self.paths = data['paths']
        self.N_runs, _, self.N_T_data = self.paths.shape
        if not self.dt == dt:
            raise ValueError('Data timestep doesnt match simulation timestep')
        
        if self.N_T > self.N_T_data:
            raise ValueError('Simulation has more timesteps than data')

    def gen_path(self):
        """
        Generate a path from the data
        """
        q = np.random.randint(self.N_runs)
        st = np.random.randint(self.N_T_data - self.N_T)
        return self.paths[q, :, st:(st + self.N_T)]

    def path_mode(self):
        return 'Experimental_data'
