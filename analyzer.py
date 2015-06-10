import numpy as np
import cPickle as pkl
import sys
import os
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

def inner_product(p1, l1x, l1y, 
                  p2, l2x, l2y, var):
    """
    Calculate the inner product between two images with the representation:
    p1 -> array of pixels
    l1x -> array of pixel x coordinate
    l2x -> array of pixel y coordinate
    each pixel is surrounded by a gaussian with variance var
    """
    N = l1x.shape[0]
    l1x = l1x.reshape(N, 1)
    l2x = l2x.reshape(1, N)

    l1y = l1y.reshape(N, 1)
    l2y = l2y.reshape(1, N)

    coupling = np.exp(-( (l1x - l2x) ** 2 + 
                         (l1y - l2y) ** 2) / (4 * var))

    return np.einsum('i,j,ij->', p1, p2, coupling)


def SNR(p1, l1x, l1y, p2, l2x, l2y, var):
    """
    Using the inner product defined above, calculates the SNR
        between two images given in sum of gaussian form
    See inner product for definitinos of variables
    Note the first set of pixels and pixel locations is 
        considered to be the ground truth
    """
    ip12 = inner_product(p1, l1x, l1y, p2, l2x, l2y, var)            
    ip11 = inner_product(p1, l1x, l1y, p1, l1x, l1y, var)
    ip22 = inner_product(p2, l2x, l2y, p2, l2x, l2y, var)

    return ip11 / (ip11 + ip22 - 2 * ip12)



class DataAnalyzer:
    def __init__(self, data):
        """
        data - dictionary containing information about run
        Loads in data from file and saves parameters in class
        """
        
        self.data = data

        self.DT = self.data['DT']
        self.N_T = self.data['N_T']

        self.XR = self.data['XR'][0]
        self.YR = self.data['YR'][0]
        self.S_gen = self.data['S_gen']
        self.N_itr = self.data['N_itr']
        self.Var = self.data['Var'][0]
        self.DC = self.data['DC']
        self.L_I = self.data['L_I']
        self.LAMBDA = self.data['LAMBDA']

        # Convert retinal positions to grid
        XS = self.data['XS']
        YS = self.data['YS']

        XS, YS = np.meshgrid(XS, YS)
        self.XS = XS.ravel()
        self.YS = YS.ravel()

        self.N_itr = self.data['N_itr']
        self.var = self.data['Var'][0]

    @classmethod    
    def fromfilename(cls, filename):
        """
        filename - filename containing data file from EM run
        Initialize the class using a filename to a pkl file
            that contains the datadict
        """
        data = pkl.load(open(filename, 'rb'))
        return cls(data)

    def SNR_idx(self, q):
        """
        Calculates the SNR between the ground truth and the 
            image produced after iteration q of the EM
        Note that we shift the image estimate by the average
            amount that the path estimate was off the true path
            (There is a degeneracy in the representation that this
            fixes. )
        """
        XYR_est = self.data['EM_data'][q]['path_means']
        S_est = self.data['EM_data'][q]['image_est']
        t = self.data['EM_data'][q]['time_steps']

        XR_est = XYR_est[:, 0]
        YR_est = XYR_est[:, 1]


        dx = np.mean(self.XR[0:t] - XR_est[0:t])
        dy = np.mean(self.YR[0:t] - YR_est[0:t])
        
        self.dx = dx
        self.dy = dy

        return SNR(self.S_gen.ravel(), self.XS, self.YS, 
                   S_est.ravel(), self.XS + dx, self.YS + dy, 
                   self.var)

    def SNR_list(self):
        """
        Returns a list giving the SNR after each iteration
        """
        return [self.SNR_idx(q) for q in range(self.N_itr)]


    def plot_path_estimate(self, q, d):
        """
        Plot the actual and estimated path generated from EM on iteration q
        d - dimension to plot (either 0 or 1)
        """
        est_mean = self.data['EM_data'][q]['path_means']
        est_sdev = self.data['EM_data'][q]['path_sdevs']
        
        if (d == 0):
            path = self.XR
            label = 'Hor.'
            dxy = self.dx
        elif (d == 1):
            path = self.YR
            label = 'Ver.'
            dxy = self.dy
        else:
            raise ValueError('d must be either 0 or 1')
        
        t = self.data['EM_data'][q]['time_steps']

        plt.fill_between(self.DT * np.arange(self.N_T), 
                         est_mean[:, d] - est_sdev[:, d], 
                         est_mean[:, d] + est_sdev[:, d], 
                         alpha = 0.5, linewidth = 1.)
        plt.plot(self.DT * np.arange(self.N_T), 
                 est_mean[:, d], label = 'estimate')
        plt.plot(self.DT * np.arange(self.N_T), 
                 path, label = 'actual')
        plt.xlabel('Time (s)')
        plt.ylabel('Relative position (pixels)')
        #plt.legend()
        plt.title(label + ' Pos., shift = %.2f' % dxy)

    def plot_EM_estimate(self, q):
        """
        Creates a full plot of the estimated image, actual image, 
            estimate and actual paths
        
        q - index of EM iteration to plot
        """
        
        if (q >= self.N_itr):
            raise ValueError('Iteration index, q, is too large')
        
        m1 = np.min(self.S_gen)
        m2 = np.max(self.S_gen)
    
        vmin = -0.1 * (m2 - m1) + m1
        vmax = m2 + 0.1 * (m2 - m1)
    
        sdev = float(np.sqrt(self.Var))    
        dt = self.DT * self.data['EM_data'][q]['time_steps'] * 1000.

        fig = plt.figure(figsize = (10, 8))
        fig.suptitle('EM Reconstruction after t = %0.2f ms for D = %0.2f, LAMBDA = %0.2f' % (dt, self.DC, self.LAMBDA))
        
        # Note actual image is convolved with a gaussian during the simulation 
        #   even though the image saved has not has this happen yet


        plt.subplot(2, 2, 1)
        plt.title('Estimated Image, S = DA: SNR = %.2f' 
                  % self.SNR_idx(q))
        S_est = self.data['EM_data'][q]['image_est']

        plt.imshow(gaussian_filter(S_est, sdev), 
                   cmap = plt.cm.gray, interpolation = 'nearest', 
                   vmin = vmin, vmax = vmax)
        plt.colorbar()


        plt.subplot(2, 2, 2)
        plt.title('Actual Image')
        plt.imshow(gaussian_filter(self.S_gen, sdev), 
                   cmap = plt.cm.gray, interpolation = 'nearest',
                   vmin = vmin, vmax = vmax)
        plt.colorbar()
    
        plt.subplot(2, 2, 3)
        self.plot_path_estimate(q, 0)

        plt.subplot(2, 2, 4)
        self.plot_path_estimate(q, 1)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

    def save_images(self):
        for q in range(self.N_itr):
            plt.clf()
            self.plot_EM_estimate(q)
            plt.savefig('img%d.png' % (100 + q))
            

if __name__ == '__main__':
    dir = sys.argv[1]
    filenames = os.listdir(dir)

    for filename in filenames:

        da = DataAnalyzer.fromfilename(dir + filename)
        SNRs = da.SNR_list()
        print SNRs
        da.plot_EM_estimate(8)
        plt.show()
#plt.plot(SNRs)
#plt.show()