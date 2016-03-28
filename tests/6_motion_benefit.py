"""
Script to show the benefit of eye motions.

(1) Spikes generated by no motion.
(2) Spikes generated including motion.

See which case it is easier to reconstruct the image.
"""

import numpy as np
from scipy.io import loadmat

from itertools import product

from src.model import EMBurak
from utils.image_gen import ImageGenerator
output_dir = 'motion_benefit1'
n_t = 200
n_repeats = 1

# Sparse coding dictionary prior
data = loadmat('sparse_coder/output/mnist_dictionary.mat')
D = data['D']

# Initalize image
_, N_pix = D.shape
L_I = int(np.sqrt(N_pix))  # Linear dimension of image

ig = ImageGenerator(L_I)
ig.make_digit()
ig.normalize()


d_ = [0.001, 100.]
motion_info_ = [
    ({'mode': 'Diffusion', 'dc': d},
     {'mode': 'Experiment', 'fpath': 'data/paths.mat'}) for d in d_]

ds_ = [0.57]  # [0.5, 0.75, 1.]
de = 1.09

for (motion_gen, motion_prior), ds in product(motion_info_, ds_):
    emb = EMBurak(
        ig.img, D, motion_gen, motion_prior, n_t=n_t, save_mode=True,
        s_gen_name=ig.img_name, ds=ds, neuron_layout='hex',
        de=de, l_n=6, n_itr=100, lamb=0.0, tau=0.05,
        output_dir_base=output_dir)
    for _ in range(n_repeats):
        XR, YR, R = emb.gen_data(ig.img, print_mode=False)
        emb.run_em(R)
        emb.save()
        emb.reset()


# FIXME: Plot the neuron grid


# convert -set delay 30 -colorspace GRAY
# -colors 256 -dispose 1 -loop 0 -scale 50% *.png alg_performance.gif

# convert -set delay 30 -colors 256
# -dispose 1 -loop 0 *.jpg alg_performance.gif
