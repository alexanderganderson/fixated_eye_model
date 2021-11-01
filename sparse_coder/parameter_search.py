import os
import h5py
from argparse import ArgumentParser
from itertools import product
import numpy as np
from scipy.io import loadmat
from sparse_coder import SparseCoder, get_effective_dimensionality
from prep_field_dataset import get_data_matrix

parser = ArgumentParser('Trains a set of Dictionaries')

parser.add_argument('--l_patch', type=int, default=32,
                    help='Size of patches.')
parser.add_argument('--output_dir', type=str, default='output/',
                    help='Base path for saving dictionaries.')
parser.add_argument('--n_itr', type=int, default=1000,
                    help='Number of iterations')
parser.add_argument('--n_bat', type=int, default=100,
                    help='Batch size.')
args = parser.parse_args()

#  data = get_data_matrix(l_patch=args.l_patch)
with h5py.File('data/final/new_extracted_patches1.h5') as f:
    assert args.l_patch == f['l_patch'].value
    data = f['white_patches'].value
    #  data = f['white_patches'][0:10000]

n_patches = data.shape[0]
data = data.reshape((n_patches, -1))

d_eff = get_effective_dimensionality(data[0:10000])
print('Effective dimensionality {}'.format(d_eff))

#  alpha_ = [0.01, 0.015, 0.02]
alpha_ = [0.015, 0.02]
over_comp_ = [2, 3.]

for alpha, over_comp in product(alpha_, over_comp_):
    n_sp = int(over_comp * d_eff)
    sc = SparseCoder(
        data=data,
        n_sp=n_sp,
        alpha=alpha,
        n_bat=args.n_bat,
        d_scale=np.std(data).astype('float32'),
        sparsify_dictionary=True
    )

    n_itr = args.n_itr
    scheme = (
        [[n_itr, 0.05]] +
        2 * [[n_itr, 0.02]] +
        3 * [[n_itr, 0.01]] +
        3 * [[n_itr, 0.005]] +
        3 * [[n_itr, 0.002]]
    )
    for n_itr, eta in scheme:
        cost_list=[]
        i_idx = sc.train(n_itr=n_itr, eta=eta, cost_list=cost_list, n_g_itr=100)

        path = os.path.join(
            args.output_dir,
            'vh_sparse_coder1_alpha_{:.3f}_overcomp_{:0.2f}_l_patch_{}.pkl'.format(
                alpha, over_comp, args.l_patch))
        print 'Saving to {}'.format(path)
        sc.save(path)
