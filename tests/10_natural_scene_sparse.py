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
from sparse_coder.sparse_coder import _get_pca
from src.model import EMBurak


#  if True:
if False:
    n_t = 5
    n_repeats = 1
    n_g_itr = 1
    output_dir = 'test'
else:
    n_t = 600
    n_repeats = 3
    n_g_itr = 320
    output_dir = 'natural_sparsity_van_hateren_pca2'

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
    10,  11,  53,  54,  61,  62,  68,  69,  96,  149, 150, 151, 181, 193, 194,
    195, 226, 234, 291, 318, 336, 352, 362, 427, 428, 498, 499, 502, 513, 532,
    533, 555, 577, 628, 629, 658, 717, 718, 750, 789, 828, 848, 856, 869, 936,
    937, 951, 973, 985, 989])

img_idx = img_idx[0:n_repeats]


normalize = lambda x: x / abs(x).max(axis=(1, 2), keepdims=True) * 0.5

data = normalize(data)
N_imgs, L_I, L_I = data.shape


def edge_filter():
    def linear_filter(x, eps=2 * 5./32):
        return (abs(x) < (1-eps)) * 1 + (abs(x) > (1 - eps)) * ((1-abs(x))/eps)
    xx = np.arange(-1, 1.001, 2./31)
    fx = fy = linear_filter(xx)
    return np.outer(fx, fy)


f = edge_filter()


# PCA Basis (capturing 90 percent of the variance)
#  D2 = get_pca_basis(data.reshape(n_imgs, -1), d_eff=d_eff)
evals, evecs = _get_pca(data.reshape(n_imgs, -1))
quad_reg = evals[::-1] ** -1
D2 = evecs.T[::-1]
quad_reg = quad_reg[0:800]
D2 = D2[0:800]

quad_reg_mean = data.reshape(n_imgs, -1).dot(D2.T).mean(axis=0)


print D2.shape


# Independent Pixel Prior
D0 = np.eye((L_I ** 2))
print D0.shape


# Desired Functionality:
# -- Optimize over different values of the sparsity penalty
# -- Look at different values of the quad_reg
# -- Look at different values of the parameters


for ds, (D, D_name, lamb, quad_reg, quad_reg_mean), dc in product(
    [0.75],
    [
        #  [D1, 'Sparse', 0.010, None, None],
        #  [D1, 'Sparse', 0.020, None, None],
        #  [D1, 'Sparse', 0.005, None, None],
        #  [D1, 'Sparse', 0.000, None, None],
        #  [D0, 'Indep',  0.000, None, None],
        [D2, 'PCA',    0.000, quad_reg, quad_reg_mean],
    ],
    #  [20., 40., 100.],
    [20.],
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
            s_range='sym',
            quad_reg=quad_reg,
            quad_reg_mean=quad_reg_mean,
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
