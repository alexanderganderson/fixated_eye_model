import numpy as np
import cPickle as pkl
import matplotlib.pyplot as plt

data_dir = 'output/'

data = pkl.load(open(data_dir + 'data.pkl', 'rb'))

XR = data['XR'][0]
YR = data['YR'][0]
S = data['S']

# Convert retinal positions to grid
XS = data['XS']
YS = data['YS']

XS, YS = np.meshgrid(XS, YS)
XS = XS.ravel()
YS = YS.ravel()

N_itr = data['N_itr']
var = data['Var'][0]


def calc_dx_dy(q):
    """
    Find dx, dy corresponding to iteration q
    """
    



def inner_product(p1, l1x, l1y, 
                  p2, l2x, l2y, var):
    """
    Calculate the inner product between two images with the representation:
    p1 -> pixels
    l1 -> location of pixels
    each pixel is surrounded by a gaussian with variance var
    """
    N = l1x.shape[0]
    l1x = l1x.reshape(N, 1)
    l2x = l2x.reshape(1, N)
    
    l1y = l1y.reshape(N, 1)
    l2y = l2y.reshape(1, N)
    
    coupling = np.exp(-( (l1x - l2x) ** 2 + (l1y - l2y) ** 2) / (4 * var))
    
    return np.einsum('i,j,ij->', p1, p2, coupling)
    
    #return coupling

def SNR(p1, l1x, l1y, p2, l2x, l2y, var):
    ip12 = inner_product(p1, l1x, l1y, p2, l2x, l2y, var)            
    ip11 = inner_product(p1, l1x, l1y, p1, l1x, l1y, var)
    ip22 = inner_product(p2, l2x, l2y, p2, l2x, l2y, var)

    return ip11 / (ip11 + ip22 - 2 * ip12)

def SNR_idx(q):
    XY_est = data['EM_data'][q]['path_means']
    S_est = data['EM_data'][q]['image_est']
    t = data['EM_data'][q]['time_steps']

    XR_est = XY_est[:, 0]
    YR_est = XY_est[:, 1]


    dx = np.mean(XR[0:t] - XR_est)
    dy = np.mean(YR[0:t] - YR_est)
    

    return SNR(S.ravel(), XS, YS, S_est.ravel(), XS + dx, YS + dy, var)

SNRs = [SNR_idx(i) for i in range(N_itr)]

plt.plot(SNRs)
plt.show()