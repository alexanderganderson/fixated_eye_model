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
from sparse_coder.sparse_coder import get_pca_basis


from src.model import EMBurak

output_dir = 'natural_sparsity_large2'
#  output_dir = 'test'
n_t = 250
n_repeats = 2
n_itr = n_t / 2
l_patch = 30

# Sparse coding dictionary prior
alpha = 0.05
over_comp = 2
path = ('sparse_coder/output/'
        'sparse_coder_alpha_{:.2f}_overcomp_{:0.2f}_l_patch_{}.pkl').format(
        alpha, over_comp, l_patch)

with open(path, 'r') as f:
    out = pkl.load(f)
    D1 = out['D']
print D1.shape
data = get_data_matrix('sparse_coder/data/final/IMAGES.npy', l_patch=l_patch)

#  data = np.random.randn(1000, l_patch ** 2)


def normalize(x, smin, smax):
    xmin, xmax = [getattr(x, func)(axis=1, keepdims=True) for func in
                  ['min', 'max']]
    u = (x - xmin) / (xmax-xmin)
    return u * (smax - smin) + smin

data = normalize(data, smin=-0.5, smax=0.5)

N_imgs, N_pix = data.shape
L_I = int(np.sqrt(N_pix))  # Linear dimension of image

# PCA Basis (capturing 90 percent of the variance)

D2 = get_pca_basis(data)
print D2.shape

# Independent Pixel Prior
D0 = np.eye((L_I ** 2))
print D0.shape


dc = 0.001
motion_gen = {'mode': 'Diffusion', 'dc': dc}
motion_prior = {'mode': 'PositionDiffusion', 'dc': dc}

#  img_idx = np.random.randint(0, N_imgs, size=n_repeats)
stds = data.std(axis=1)
pct = np.percentile(stds, 99)
img_idx = np.where(stds > pct)[0]
np.random.shuffle(img_idx)
assert len(img_idx) > n_repeats
img_idx = img_idx[0:n_repeats]


for ds, (D, D_name, lamb) in product(
    [0.5, 0.75, 1.0],
    #  zip([D2,
    #       D1,
    #       D0],
    #      ['PCA',
    #       'Sparse',
    #       'Indep'],
    #      [0.0,
    #       0.01,
    #       0.0])
    #  zip([D0], ['Indep'], [0.0]),
    zip([D2], ['PCA'], [0.0])
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
        lamb=lamb,
        tau=1.28,
        n_g_itr=320,
        #  n_g_itr=10,
        output_dir_base=output_dir,
        s_range='sym'
    )

    for u in img_idx:
        XR, YR, R = emb.gen_data(data[u].reshape(L_I, L_I))
        #  XR, YR, R = emb.gen_data(np.zeros((L_I, L_I)))
        emb.data['D_name'] = D_name
        emb.run_em(R)
        emb.save()
        emb.reset()


# convert -set delay 30 -colorspace GRAY
# -colors 256 -dispose 1 -loop 0 -scale 50% *.png alg_performance.gif

# convert -set delay 30 -colors 256
# -dispose 1 -loop 0 *.jpg alg_performance.gif
