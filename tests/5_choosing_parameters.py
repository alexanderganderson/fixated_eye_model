"""Script to choose the parameters of the model."""

# 5_choosing_parameters.py
# Script for choosing parameters for model

# Parameters to check:
# gamma - pixel out of bounds parameter
# lamb - sparsity prior
# fista_c error on fista 2nd derivative uppper bound
# n_g_itr - number of gradient steps
# n_itr - number of chunks of optimization
#     Seems roughly equivalent... slightly better for more
# tau - decay constant for summing hessian
# n_p - number of particles for M

from os.path import join

import numpy as np
from scipy.io import loadmat

from src.model import EMBurak
from utils.image_gen import ImageGenerator

output_dir = 'parameter_cv'

n_repeats = 20

data = loadmat('sparse_coder/output/mnist_dictionary.mat')
D = data['D']

_, N_pix = D.shape

L_I = int(np.sqrt(N_pix))  # Linear dimension of image

D0 = np.eye((L_I ** 2))


ig = ImageGenerator(L_I)
ig.make_digit()
ig.normalize()

s_gen = ig.img
s_gen_name = ig.img_name

n_t = 100
LAMBDA = 0.0


# motion_gen = {'mode': 'Experiment', 'fpath': 'data/paths.mat'}
motion_gen = {'mode': 'Diffusion', 'dc': 20.}
motion_prior = {'mode': 'PositionDiffusion', 'dc': 20.}

gamma_ = [1., 10., 100.]
lamb_ = [0., 0.01, 0.1]
fista_c_ = [0.3, 0.8, 1.5]
n_g_itr_ = [20, 40, 60]
n_itr_ = [10, 20, 50]
tau_ = [0.05, 0.2, 0.5]
n_p_ = [5, 20, 40]

param__ = [gamma_, lamb_, fista_c_, n_g_itr_, n_itr_, tau_, n_p_]
param_name_ = ['gamma', 'lamb', 'fista_c', 'n_g_itr', 'n_itr', 'tau', 'n_p']
for param_, param_name in zip(param__, param_name_):
    output_dir = join('parameter_cv', param_name)
    for param in param_:
        emb = EMBurak(
            s_gen, D, motion_gen, motion_prior, n_t=n_t, save_mode=True,
            s_gen_name=s_gen_name, ds=0.57, neuron_layout='hex',
            de=1.09, l_n=6, print_mode=False, output_dir_base=output_dir,
            **{param_name: param})
        for _ in range(n_repeats):
            XR, YR, R = emb.gen_data(s_gen)
            emb.run_em(R)
            emb.save()
            emb.reset()
