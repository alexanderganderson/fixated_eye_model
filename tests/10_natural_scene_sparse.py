"""
Code to show the usefulness of a sparse prior on MNIST.

Compares:
(1) Sparse Prior
(2) Dictionary with zero sparsity (like PCA, but positive only, variance not
    ordered)
(3) Independent pixel prior
"""

import numpy as np
import pickle as pkl
from itertools import product
from sparse_coder.prep_field_dataset import get_data_matrix


from src.model import EMBurak

output_dir = 'natural_sparsity'
n_t = 200
n_repeats = 20
n_itr = n_t / 2
l_patch=16

# Sparse coding dictionary prior
alpha = 0.1
over_comp = 2
path = 'sparse_coder/output/sparse_coder_alpha_{:.2f}_overcomp_{:0.2f}.pkl'.format(
        alpha, over_comp)

with open(path, 'r') as f:
    out = pkl.load(f)
    D1 = out['D']
print D1.shape
data = get_data_matrix('sparse_coder/data/final/IMAGES.npy', l_patch=l_patch)

def normalize(x, smin, smax):
    xmin, xmax = [getattr(x, func)(axis=1, keepdims=True) for func in
                  ['min', 'max']]
    u = (x - xmin) / (xmax-xmin)
    return u * (smax - smin) + smin

data = normalize(data, smin=-0.5, smax=0.5)

# Initalize image
N_imgs, N_pix = data.shape
L_I = int(np.sqrt(N_pix))  # Linear dimension of image


# Independent Pixel Prior
D0 = np.eye((L_I ** 2))
print D0.shape

motion_gen = {'mode': 'Diffusion', 'dc': 100.}
motion_prior = {'mode': 'PositionDiffusion', 'dc': 100.}

for ds, (D, D_name) in product(
    [0.5, 0.7, 1.0],
    zip([D0, D1],
        ['Indep', 'Sparse'])
):
        emb = EMBurak(
            np.zeros((L_I, L_I)),
            D,
            motion_gen,
            motion_prior,
            n_t=n_t,
            save_mode=True,
            s_gen_name='natural_image',
            ds=ds,
            neuron_layout='sqr',
            de=1.,
            l_n=18,
            n_itr=n_itr,
            lamb=0.0,
            tau=1.28,
            n_g_itr=320,
            output_dir_base=output_dir,
            s_range='sym'
        )
        for _ in range(n_repeats):
            u = np.random.randint(N_imgs)
            #  XR, YR, R = emb.gen_data(data[u].reshape(L_I, L_I))
            XR, YR, R = emb.gen_data(np.zeros((L_I, L_I)))
            emb.data['D_name'] = D_name
            emb.run_em(R)
            emb.save()
            emb.reset()


# convert -set delay 30 -colorspace GRAY
# -colors 256 -dispose 1 -loop 0 -scale 50% *.png alg_performance.gif

# convert -set delay 30 -colors 256
# -dispose 1 -loop 0 *.jpg alg_performance.gif
