import numpy as np
import cPickle as pkl
import sys
#import matplotlib.pyplot as plt

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
    def __init__(self, filename):
        """
        filename - filename containing data file from EM run
        Loads in data from file and saves parameters in class
        """
        
        self.data = pkl.load(open(filename, 'rb'))

        self.XR = self.data['XR'][0]
        self.YR = self.data['YR'][0]
        self.S = self.data['S']

        # Convert retinal positions to grid
        XS = self.data['XS']
        YS = self.data['YS']

        XS, YS = np.meshgrid(XS, YS)
        self.XS = XS.ravel()
        self.YS = YS.ravel()

        self.N_itr = self.data['N_itr']
        self.var = self.data['Var'][0]

    

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


        dx = np.mean(self.XR[0:t] - XR_est)
        dy = np.mean(self.YR[0:t] - YR_est)
    

        return SNR(self.S.ravel(), self.XS, self.YS, 
                   S_est.ravel(), self.XS + dx, self.YS + dy, 
                   self.var)


if __name__ == '__main__':
    da = DataAnalyzer(sys.argv[1])

    print [da.SNR_idx(i) for i in range(da.N_itr)]

#plt.plot(SNRs)
#plt.show()