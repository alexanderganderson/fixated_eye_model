import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'nearest'

from src.gauss import EMGauss
from utils.image_gen import ImageGenerator
from sparse_coder.prep_field_dataset import get_data_matrix


l_i = 20
ds = 0.3
de = 1.09
rf_ratio = 0.203
# rf_ratio = 0.4
n_t = 75

l_n = (l_i * ds / np.sqrt(2))

mat = get_data_matrix(path='sparse_coder/data/final/IMAGES.npy', l_patch=l_i,
                      n_patches=10)

s_gen = mat[3].astype('float32')
s_gen = np.clip(s_gen, -1, 1)



emg = EMGauss(
    l_i=l_i,
    motion_gen={'mode': 'Diffusion', 'dc': 4.},
    motion_prior={'dc': 10.},
    n_t=n_t,
    ds=ds,
    de=de,
    n_p=50,
    print_mode=True,
    l_n=l_n,
    rf_ratio=rf_ratio,
    sig_obs=0.1
)

m, xr, yr = emg.gen_data(s_gen)

s, data = emg.run_em(m, n_passes=1, n_itr=n_t, reg=1.)
emg.pf.calculate_means_sdevs()
means = emg.pf.means
sdevs = emg.pf.sdevs


with emg.tb.sess.as_default():
    xe, ye = emg.tb.sess.run([emg.tb.t_xe, emg.tb.t_ye])
    xs, ys = emg.tb.sess.run([emg.tb.t_xs, emg.tb.t_ys])





