import numpy as np
import matplotlib.pyplot as plt

class Center:
    def __init__(self, _Lx, _D, _dt):
        """
        Class that Implements a diffusing center in a box of size Lx
        
        Lx = linear dimension of square to diffuse in
        D = diffusion constant
        dt = timestep size
        x = coordinates of the current location of the random walk, 
            initialized as [0, 0]
        Initializes Center Object
        """
        self.Lx = _Lx
        self.D = _D
        self.m0 = np.array([0, 0], dtype = 'float64') # current position
        self.dt = _dt
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
            temp = self.mn + np.random.normal(size = 2, scale = self.sn)
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