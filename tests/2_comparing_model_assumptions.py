"""
The goal of this script is to run the model using different assumptions
and to then compare the results
"""

import numpy as np
from scipy.io import loadmat

from src.model import EMBurak
from utils.image_gen import ImageGenerator

output_dir = 'comparing_motion_priors'

data = loadmat('sparse_coder/output/mnist_dictionary.mat')
D1 = data['D']

_, N_pix = D1.shape

L_I = int(np.sqrt(N_pix))  # Linear dimension of image

D0 = np.eye((L_I ** 2))


ig = ImageGenerator(L_I)
# ig.make_digit(mode = 'random')
ig.make_digit()
ig.normalize()

s_gen = ig.img
s_gen_name = ig.img_name

n_t = 50
LAMBDA = 0.0

# motion_gen0 = {'mode': 'Diffusion', 'dc': 100.}
motion_gen = {'mode': 'Experiment', 'fpath': 'data/paths.mat'}


motion_prior0 = {'mode': 'PositionDiffusion', 'dc': 0.00001}
motion_prior1 = {'mode': 'PositionDiffusion', 'dc': 20.}
motion_prior2 = {'mode': 'VelocityDiffusion',
                 'v0': np.sqrt(40.), 'dcv': 100.}

modes = []
# Motion Prior: no motion, Image Prior: Independent Pixels
# modes.append((motion_prior0, D0))

# Motion Prior: diffusion, Image Prior: Independent Pixels
# modes.append((motion_prior1, D0))

# Motion Prior: diffusion, Image Prior: MNIST
# modes.append((motion_prior1, D1))

# Motion Prior: velocity diffusion, Independent Pixels
# modes.append((motion_prior2, D0))

# Motion Prior: velocity diffusion, Image Prior MNIST
modes.append((motion_prior2, D1))

# FIXME: what to choose to ds

for motion_prior, D in modes:
    emb = EMBurak(
        s_gen, D, motion_gen, motion_prior, n_t=n_t, save_mode=True,
        s_gen_name=s_gen_name, ds=0.57, neuron_layout='hex',
        de=1.09, l_n=6, n_itr=10, lamb=0.0, tau=0.05)
    for _ in range(10):
        XR, YR, R = emb.gen_data(s_gen, print_mode=False)
        emb.run_EM(R)
        emb.save()
        emb.reset()
