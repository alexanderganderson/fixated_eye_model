import numpy as np
import matplotlib.pyplot as plt
import cPickle as pkl

def plot_path(XY_act, est_mean, est_sdev, d):
    DT = 0.005
    N_T = XY_act.shape[0]

    plt.fill_between(DT * np.arange(N_T), 
                     est_mean[:, d] - est_sdev[:, d], 
                     est_mean[:, d] + est_sdev[:, d], 
                     alpha = 0.5, linewidth = 1.)
    plt.plot(DT * np.arange(N_T), est_mean[:, d], label = 'estimate')
    plt.plot(DT * np.arange(N_T), XY_act[:, d], label = 'actual')
    plt.xlabel('Time (s)')
    plt.ylabel('Relative position (pixels)')
    plt.legend()
    plt.title('Estimated position for dim' + str(d))


def plot_image_estimate(S_actual, S_est, XY_act, est_mean, est_sdev):
    """
    Plots of the estimated and actual images
    S_actual - actual image
    S_est - estimated image
    """
    
    vmin = -0.1
    vmax = 1 + 0.1
    
    
    
    #L_I = S_actual.shape[0]
    N_T = XY_act.shape[0]
    
    plt.subplot(2, 3, 1)
    plt.title('Estimate')
    plt.imshow(S_est, 
               cmap = plt.cm.gray, 
               interpolation = 'nearest', 
               vmin = vmin, vmax = vmax)
    plt.colorbar()
    
    plt.subplot(2, 3, 2)
    plt.title('Actual')
    plt.imshow(S_actual, cmap = plt.cm.gray, 
               interpolation = 'nearest',
               vmin = vmin, vmax = vmax)
    plt.colorbar()
    plt.subplot(2, 3, 3)
    plt.title('Error')
    plt.imshow(np.abs(S_actual - S_est), 
               cmap = plt.cm.gray, interpolation = 'nearest')
    plt.colorbar()
    
    
    plt.subplot(2, 3, 4)
    plot_path(XY_act, est_mean, est_sdev, 0)


    
    plt.subplot(2, 3, 5)
    plot_path(XY_act, est_mean, est_sdev, 1)
    
    plt.show()

f1 = 'images.pkl'
f2 = 'paths.pkl'

images = pkl.load(open(f1, 'rb'))
paths = pkl.load(open(f2, 'rb'))

S_actual = images['truth']



N_T = paths['truthX'].shape[0]

S_est = images[(19, N_T)]

XY_act = np.zeros((N_T, 2))
XY_act[:, 0] = paths['truthX']
XY_act[:, 1] = paths['truthY']

est_mean = paths[(19, 300, 'means')]
est_sdevs = paths[(19, 300, 'sdevs')]

plot_image_estimate(S_actual, S_est, XY_act, est_mean, est_sdevs)
