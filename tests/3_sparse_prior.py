"""
Code to show the usefulness of a sparse prior on MNIST.

Compares:
(1) Sparse Prior
(2) Dictionary with zero sparsity (like PCA, but positive only, variance not
    ordered)
(3) Independent pixel prior
"""

import numpy as np
from scipy.io import loadmat

from src.model import EMBurak
from utils.image_gen import ImageGenerator
output_dir = 'sparsity'
n_t = 200
n_repeats = 20
n_itr = n_t / 2


# Sparse coding dictionary prior
data1 = loadmat('sparse_coder/output/mnist_dictionary.mat')
D1 = data1['D']

# Positive only code with no sparsity during dictionary training
data2 = loadmat('sparse_coder/output/mnist_dictionary_not_sparse.mat')
D2 = data2['D']

# Initalize image
_, N_pix = D1.shape
L_I = int(np.sqrt(N_pix))  # Linear dimension of image

ig = ImageGenerator(L_I)
ig.make_digit()
ig.normalize()

# Independent Pixel Prior
D0 = np.eye((L_I ** 2))


motion_gen = {'mode': 'Diffusion', 'dc': 100.}
motion_prior = {'mode': 'PositionDiffusion', 'dc': 100.}

for ds in [0.5, 0.7, 1.0]:
    for D, D_name in zip([D0, D1, D2], ['Indep', 'Sparse', 'Non-sparse']):
        emb = EMBurak(
            ig.img, D, motion_gen, motion_prior, n_t=n_t, save_mode=True,
            s_gen_name=ig.img_name, ds=ds, neuron_layout='sqr',
            de=1., l_n=18, n_itr=n_itr, lamb=0.0, tau=1.28, n_g_itr=320,
            output_dir_base=output_dir)
        for _ in range(n_repeats):
            XR, YR, R = emb.gen_data(ig.img)
            emb.data['D_name'] = D_name
            emb.run_em(R)
            emb.save()
            emb.reset()


# convert -set delay 30 -colorspace GRAY
# -colors 256 -dispose 1 -loop 0 -scale 50% *.png alg_performance.gif

# convert -set delay 30 -colors 256
# -dispose 1 -loop 0 *.jpg alg_performance.gif
