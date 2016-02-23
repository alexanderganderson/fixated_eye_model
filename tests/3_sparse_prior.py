"""
Code to compare different prior strengths
"""

import numpy as np
from scipy.io import loadmat

from src.model import EMBurak
from utils.image_gen import ImageGenerator
output_dir = 'sparsity'
n_t = 200
n_repeats = 1

# Sparse coding dictionary prior
data = loadmat('sparse_coder/output/mnist_dictionary.mat')
D1 = data['D']

_, N_pix = D1.shape
L_I = int(np.sqrt(N_pix))  # Linear dimension of image

ig = ImageGenerator(L_I)
ig.make_digit()
ig.normalize()

# Independent Pixel Prior
D0 = np.eye((L_I ** 2))


motion_gen = {'mode': 'Diffusion', 'dc': 100.}
motion_prior = {'mode': 'PositionDiffusion', 'dc': 100.}

for D in [D0, D1]:
    emb = EMBurak(
        ig.img, D, motion_gen, motion_prior, n_t=n_t, save_mode=True,
        s_gen_name=ig.img_name, ds=1., neuron_layout='sqr',
        de=1., l_n=18, n_itr=100, lamb=0.0, tau=0.05,
        output_dir_base=output_dir)
    for _ in range(n_repeats):
        XR, YR, R = emb.gen_data(ig.img, print_mode=False)
        emb.run_EM(R)
        emb.save()
        emb.reset()


# convert -set delay 30 -colorspace GRAY -colors 256 -dispose 1 -loop 0 -scale 50% *.png alg_performance.gif

# convert -set delay 30 -colors 256 -dispose 1 -loop 0 *.jpg alg_performance.gif
