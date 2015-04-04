import numpy as np
import matplotlib.pyplot as plt
import cPickle as pkl
from scipy.ndimage.filters import gaussian_filter
from utils.SNR import SNR

def plot_path(XY_act, est_mean, est_sdev, d, params):
    """
    Plot the actual and estimated path generated from EM
    
    XY_act - matrix giving actual path for X,Y
    est_mean - mean of estimate of path
    est_sdev - sdev of estimate of path
    d - dimension to plot (either 0 or 1)
    params - dictionary of parameters
    """
    DT = params['DT']
    N_T = params['N_T']
    t = est_mean.shape[0]
    plt.fill_between(DT * np.arange(t), 
                     est_mean[:, d] - est_sdev[:, d], 
                     est_mean[:, d] + est_sdev[:, d], 
                     alpha = 0.5, linewidth = 1.)
    plt.plot(DT * np.arange(t), est_mean[:, d], label = 'estimate')
    plt.plot(DT * np.arange(N_T), XY_act[:, d], label = 'actual')
    plt.xlabel('Time (s)')
    plt.ylabel('Relative position (pixels)')
    plt.legend()
    plt.title('Estimated position for dim' + str(d))


def plot_image_estimate(S_actual, S_est, XY_act, est_mean, est_sdev, params):
    """
    Creates a full plot of the estimated image, actual image, 
        estimate and actual paths
        
    S_actual - actual image
    S_est - estimated image
    XY_act - actual path
    est_mean - mean of estimated path
    est_sdev - sdev of estimated path
    params - dictionary of parameters
    """    
    m1 = np.min(S_actual)
    m2 = np.max(S_actual)
    
    vmin = -0.1 * (m2 - m1) + m1
    vmax = m2 + 0.1 * (m2 - m1)
    
    N_T = params['N_T']
    
    plt.subplot(2, 2, 1)
    plt.title('Estimate: SNR = ' + str(SNR(S_actual, S_est)))
    plt.imshow(S_est, cmap = plt.cm.gray, interpolation = 'nearest', 
               vmin = vmin, vmax = vmax)
    plt.colorbar()
    
    plt.subplot(2, 2, 2)
    plt.title('Actual')
    plt.imshow(S_actual, cmap = plt.cm.gray, interpolation = 'nearest',
               vmin = vmin, vmax = vmax)
    plt.colorbar()
    
    plt.subplot(2, 2, 3)
    plot_path(XY_act, est_mean, est_sdev, 0, params)

    plt.subplot(2, 2, 4)
    plot_path(XY_act, est_mean, est_sdev, 1, params)




# Code to load and 

images_fn = 'images.pkl'
paths_fn = 'paths.pkl'
params_fn = 'params.pkl'


images = pkl.load(open(images_fn, 'rb'))
paths = pkl.load(open(paths_fn, 'rb'))
params = pkl.load(open(params_fn, 'rb'))


S_actual = images['truth']
# Note actual image is convolved with a gaussian during the simulation 
#   even though the image saved has not has this happen yet
S_actual = gaussian_filter(S_actual, 0.5)

N_T = params['N_T']

XY_act = np.zeros((N_T, 2))
XY_act[:, 0] = paths['truthX']#[0:t]
XY_act[:, 1] = paths['truthY']#[0:t]


N_T = params['N_T']
N_itr = params['N_itr']
for i in range(N_itr):
    # Images are indexed by the iteration number
    S_est = images[i]
    S_est = gaussian_filter(S_est, 0.5)

    est_mean = paths[(i, 'means')]
    est_sdevs = paths[(i, 'sdevs')]

    plot_image_estimate(S_actual, S_est, XY_act, est_mean, est_sdevs, params)
    plt.savefig('img/EM_est_prior/img' + str(100 + i) + '.png', dpi = 100)
    plt.clf()

# After running, go to image directory and run
# convert -set delay 25 -dispose 1 -loop 0 *.png EM_est_D100.gif
#  to convert into a gif