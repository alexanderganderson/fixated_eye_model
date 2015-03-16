import numpy as np
import matplotlib.pyplot as plt
import cPickle as pkl

def plot_image_estimate(S_actual, S_est):
    """
    Plots of the estimated and actual images
    S_actual - actual image
    S_est - estimated image
    """
    
    vmin = -1.
    vmax = 1.
    
    #L_I = S_actual.shape[0]
    
    plt.subplot(1, 3, 1)
    plt.title('Estimate')
    plt.imshow(S_est, 
               cmap = plt.cm.gray, 
               interpolation = 'nearest', 
               vmin = vmin, vmax = vmax)
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.title('Actual')
    plt.imshow(S_actual, cmap = plt.cm.gray, 
               interpolation = 'nearest',
               vmin = vmin, vmax = vmax)
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.title('Error')
    plt.imshow(np.abs(S_actual - S_est), 
               cmap = plt.cm.gray, interpolation = 'nearest')
    plt.colorbar()
    plt.show()

f1 = 'images.pkl'
f2 = 'paths.pkl'

images = pkl.load(open(f1, 'rb'))
paths = pkl.load(open(f2, 'rb'))

S_actual = images['truth']

S_est = images[(19, 300)]

plot_image_estimate(S_actual, S_est)
