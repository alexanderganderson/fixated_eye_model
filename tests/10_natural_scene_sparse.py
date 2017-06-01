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

output_dir = 'natural_sparsity_van_hateren2'
#  output_dir = 'test'
if False:
    n_t = 5
    n_repeats = 1
    n_g_itr = 1
else:
    n_t = 600
    n_repeats = 20
    n_g_itr = 320

n_itr = n_t / 5
l_patch = 32

# Sparse coding dictionary prior
alpha = 0.015
over_comp = 3
path = ('sparse_coder/output/'
        'vh_sparse_coder_alpha_{:.3f}_overcomp_{:0.2f}_l_patch_{}.pkl').format(
        alpha, over_comp, l_patch)

with open(path, 'r') as f:
    out = pkl.load(f)
    D1 = out['D']
print D1.shape
d_eff = D1.shape[0] / over_comp



# Load images

with h5py.File('sparse_coder/data/final/new_extracted_patches.h5') as f:
    assert l_patch == f['l_patch'].value
    n_imgs = 10000
    data = f['white_patches'][0:n_imgs]


img_idx = np.array([
    434, 569, 752, 114,  78,   2, 263, 720, 986,  86, 115, 307, 385, 767, 959,
    692, 399,  92, 886, 488, 100, 606, 209, 148, 646, 600, 662, 533, 618, 860,
    427, 115, 798, 826,  48, 724, 116, 569, 307, 302, 232, 469, 688, 624, 134,
    852, 665,  74, 876, 790,  60, 246, 405, 549, 123, 938, 227, 829, 888, 438,
    353, 992, 158, 685, 843,  58, 288, 914, 289, 687, 246, 392, 443, 748,  66,
    652, 328,  47,  77, 375, 617, 468, 339, 429, 778, 141, 326, 240, 780, 400,
    951, 212,   4, 185, 671, 127, 305, 324], dtype='int32')

#  data = get_data_matrix('sparse_coder/data/final/IMAGES.npy', l_patch=l_patch)

#  data = np.random.randn(1000, l_patch ** 2)


def normalize(x):
    xmin, xmax = [getattr(x, func)(axis=(1, 2), keepdims=True) for func in
                  ['min', 'max']]
    u = x / np.maximum(-xmin, xmax)
    return u * 0.5


def normalize1(x)
    return x / abs(x).max(axis=(1, 2), keepdims=True) * 0.5

data = normalize(data)



N_imgs, L_I, L_I = data.shape
#  L_I = int(np.sqrt(N_pix))  # Linear dimension of image



# PCA Basis (capturing 90 percent of the variance)
D2 = get_pca_basis(data.reshape(n_imgs, -1), d_eff=d_eff * 2)
print D2.shape

# Independent Pixel Prior
D0 = np.eye((L_I ** 2))
print D0.shape


dc = 20.
motion_gen = {'mode': 'Diffusion', 'dc': dc}
motion_prior = {'mode': 'PositionDiffusion', 'dc': dc}

#  img_idx = np.random.randint(0, N_imgs, size=n_repeats)
#  stds = data.std(axis=1)
#  pct = np.percentile(stds, 99)
#  img_idx = np.where(stds > pct)[0]
#  np.random.shuffle(img_idx)
good_idx = [1, 2, 3, 4, 12, 13, 14, 16, 17, 18, 23, 24, 26, 28]
img_idx = img_idx[good_idx]
#  assert len(img_idx) > n_repeats
img_idx = img_idx[5:n_repeats]


for ds, (D, D_name, lamb) in product(
    [0.75],
    #  zip([
    #      #D2,
    #       D1,
    #       D0
    #      ],
    #      [#'PCA',
    #       'Sparse',
    #       'Indep'
    #      ],
    #      [#0.0,
    #       0.01,
    #       0.0
    #      ])
    #  zip([D0], ['Indep'], [0.0]),
    zip([D2], ['PCA'], [0.0])
):
    l_n = int(l_patch * ds / np.sqrt(2) +
           0.25 * 2 * np.sqrt(dc * n_t * 0.001))

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

        XR, YR, R = emb.gen_data(data[u])
        emb.data['D_name'] = D_name
        emb.run_em(R)
        emb.save()
        emb.reset()


# convert -set delay 30 -colorspace GRAY
# -colors 256 -dispose 1 -loop 0 -scale 50% *.png alg_performance.gif

# convert -set delay 30 -colors 256
# -dispose 1 -loop 0 *.jpg alg_performance.gif
