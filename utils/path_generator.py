"""Code to generate an eye path."""

import abc
import numpy as np
from scipy.io import loadmat


class Center:
    """Code modeling diffusion in a box."""

    def __init__(self, lx, dc, dt):
        """
        Class that Implements a diffusing center in a box of size Lx.

        Lx = linear dimension of square to diffuse in
        DC = diffusion constant
        DT = timestep size
        x = coordinates of the current location of the random walk,
            initialized as [0, 0]
        Initializes Center Object
        """
        self.lx = lx
        self.dc = dc
        self.m0 = np.array([0, 0], dtype='float64')  # current position
        self.dt = dt
        # The diffusion is biased towards the center by taking a product of
        # gaussians
        # A product of gaussians is also a gaussian with mean, sdev given as
        # (mn, sn)
        self.m1 = np.array([0, 0], dtype='float64')  # center of image
        # Standard deviation for diffusion
        self.s0 = np.sqrt(self.dc * self.dt)
        self.s1 = self.lx / 4  # Standard Deviation for centering gaussian
        self.sn = 1 / np.sqrt(1 / self.s0 ** 2 + 1 / self.s1 ** 2)

    def advance(self):
        """Update location according to a random walk in a box."""
        self.mn = (
            self.m0 / self.s0 ** 2 + self.m1 / self.s1 ** 2) * self.sn ** 2
        while(True):
            # Note that for 2d diffusion, each component's variance is half the
            #   variance of the overall step length
            temp = self.mn + \
                np.random.normal(size=2, scale=self.sn / np.sqrt(2))
            if (temp[0] > - self.lx / 2 and
                    temp[0] < self.lx / 2 and
                    temp[1] > - self.lx / 2 and
                    temp[1] < self.lx / 2):

                self.m0 = temp
                break

    def get_center(self):
        """Get the center."""
        return self.m0

    def reset(self):
        """Reset the center."""
        self.m0 = np.array([0, 0], dtype='float64')

    def __str__(self):
        """Return a string giving the center."""
        return str(self.m0)


class PathGenerator():
    """Abstract class describing a path generator."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, n_t, *args):
        """Initialize a 2d Path Generator."""
        self.n_t = n_t

    @abc.abstractmethod
    def gen_path(self):
        """Generate a 2d path with N_T timesteps starting at (0,0)."""
        return np.zeros((2, self.n_t))

    @abc.abstractmethod
    def mode(self):
        """Return a string describing the path."""
        return ' '


class DiffusionPathGenerator(PathGenerator):
    """Create a path generator that does diffusion."""

    def __init__(self, n_t, lx, dc, dt):
        """
        Create a path generator that does bounded diffusion.

        Note: initial point is randomly generated.

        Parameters
        ----------
        n_t : int
            Number of timesteps to generate path
        lx : int
            Window that the diffusion is restricted to
        dc : float
            Diffusion constant
        dt : float
            Timestep
        """
        PathGenerator.__init__(self, n_t)
        self.c = Center(lx, dc, dt)

    def gen_path(self):
        """Generate a path."""
        self.c.reset()
        path = np.zeros((2, self.n_t))
        for i in range(self.n_t):
            path[:, i] = self.c.get_center()
            self.c.advance()
        return path

    def mode(self):
        """Get mode name."""
        return 'Diffusion'


class ExperimentalPathGenerator(PathGenerator):
    """Class that generates paths based on experimental data."""

    def __init__(self, n_t, filename, dt):
        """
        Create a path generator that uses real experimental data.

        Parameters
        ----------
        filename : str
            filename for pkl file that contains an array of paths
              data['paths'] = (N_runs, number of timesteps, 2)
              2 - number of dimensions of path eg. x,y
        n_t : int
            Number of timesteps
        dt : float
            Time between samples for desired path.
        """
        PathGenerator.__init__(self, n_t)
        self.data = loadmat(filename)
        self.dt = self.data['DT'][0, 0]
        self.paths = self.data['paths']
        self.n_runs, self.n_t_data, _ = self.paths.shape

        if not self.dt == dt:
            raise ValueError('Data timestep doesnt match simulation timestep')
        if self.n_t > self.n_t_data:
            pass  # raise ValueError('Simulation has more timesteps than data')

    def gen_path(self):
        """Generate a path from the data."""
        q = np.random.randint(self.n_runs)
        st = np.random.randint(self.n_t_data - self.n_t)
        pat = self.paths[q, st:(st + self.n_t), :].transpose()
        pat = pat - pat[:, 0:1]
        return pat

    def mode(self):
        """Return the path generation mode."""
        return 'Experimental_data'


if __name__ == '__main__':
    pg = DiffusionPathGenerator(100, 10, 100., 0.001, 1)
    path = pg.gen_path()
    print path

    fn = 'data/paths.mat'
    pg = ExperimentalPathGenerator(100, fn, 0.001)
    path = pg.gen_path()
    print path
