"""
Code to show the usefulness of a sparse prior on natural scene patches.

Compares:
(1) Sparse Prior
(2) Dictionary with zero sparsity (like PCA, but positive only, variance not
    ordered)
(3) Independent pixel prior
"""

import numpy as np
import pickle as pkl
import h5py
from itertools import product
#  from sparse_coder.prep_field_dataset import get_data_matrix
from sparse_coder.sparse_coder import get_pca_basis
from src.model import EMBurak

output_dir = 'natural_sparsity_van_hateren3'
#  output_dir = 'test'
if False:
    n_t = 5
    n_repeats = 1
    n_g_itr = 1
else:
    n_t = 600
    n_repeats = 4
    n_g_itr = 320

n_itr = n_t / 5
l_patch = 32

# Sparse coding dictionary prior
alpha = 0.02
over_comp = 3
path = ('sparse_coder/output/'
        'vh_sparse_coder1_alpha_{:.3f}_overcomp_{:0.2f}_l_patch_{}.pkl').format(
        alpha, over_comp, l_patch)

with open(path, 'r') as f:
    out = pkl.load(f)
    D1 = out['D']
print D1.shape
d_eff = D1.shape[0] / over_comp

# Load images

with h5py.File('sparse_coder/data/final/new_extracted_patches1.h5') as f:
    assert l_patch == f['l_patch'].value
    n_imgs = 10000
    data = f['white_patches'][0:n_imgs]


img_idx = np.array([
    10,  11,  53,  54,  61,  62,  68,  69,  96, 149, 150, 151, 181, 193, 194,
    195, 226, 234, 291, 318, 336, 352, 362, 427, 428, 498, 499, 502, 513, 532,
    533, 555, 577, 628, 629, 658, 717, 718, 750, 789, 828, 848, 856, 869, 936,
    937, 951, 973, 985, 989])

img_idx = img_idx[0:n_repeats]



normalize = lambda x: x / abs(x).max(axis=(1, 2), keepdims=True) * 0.5

data = normalize(data)
N_imgs, L_I, L_I = data.shape


def edge_filter():
    def linear_filter(x, eps = 2 * 5./32):
        return (abs(x) < (1 - eps)) * 1 + (abs(x) > (1 - eps)) * ((1-abs(x))/eps)
    xx = np.arange(-1, 1.001, 2./31)
    fx = fy = linear_filter(xx)
    return np.outer(fx, fy)

f = edge_filter()


# PCA Basis (capturing 90 percent of the variance)
D2 = get_pca_basis(data.reshape(n_imgs, -1), d_eff=d_eff)
print D2.shape

# Independent Pixel Prior
D0 = np.eye((L_I ** 2))
print D0.shape

for ds, (D, D_name, lamb), dc in product(
    [0.75],
    [
        #  [D1, 'Sparse', 0.01],
        #  [D1, 'Sparse', 0.02],
        #  [D1, 'Sparse', 0.005],
        #  [D0, 'Indep', 0.0],
        [D2, 'PCA', 0.0]
    ],
    [20., 40., 100.]
):
    l_n = int(l_patch * ds / np.sqrt(2) +
           0.25 * 2 * np.sqrt(dc * n_t * 0.001))

    motion_gen = {'mode': 'Diffusion', 'dc': dc}
    motion_prior = {'mode': 'PositionDiffusion', 'dc': dc}

    for u in img_idx:
        emb = EMBurak(
            np.zeros((L_I, L_I)),
            D,
            motion_gen,
            motion_prior,
            n_t=n_t,
            save_mode=True,
            s_gen_name='van_hateren_natural_image',
            ds=ds,
            neuron_layout='hex',
            de=1.,
            l_n=l_n,
            n_itr=n_itr,
            lamb=lamb,
            tau=1.28,
            n_g_itr=n_g_itr,
            output_dir_base=output_dir,
            s_range='sym'
        )

        XR, YR, R = emb.gen_data(data[u] * f)
        emb.data['D_name'] = D_name
        emb.run_em(R)
        emb.save()
        emb.reset()


# convert -set delay 30 -colorspace GRAY
# -colors 256 -dispose 1 -loop 0 -scale 50% *.png alg_performance.gif

# convert -set delay 30 -colors 256
# -dispose 1 -loop 0 *.jpg alg_performance.gif
