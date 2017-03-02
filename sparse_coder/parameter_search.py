import os
from argparse import ArgumentParser
from itertools import product
import numpy as np
from scipy.io import loadmat
from sparse_coder import SparseCoder, get_effective_dimensionality
from prep_field_dataset import get_data_matrix

parser = ArgumentParser('Trains a set of Dictionaries')

parser.add_argument('--l_patch', type=int, default=16,
                    help='Size of patches.')
parser.add_argument('--output_dir', type=str, default='output/',
                    help='Base path for saving dictionaries.')
parser.add_argument('--n_itr', type=int, default=3000,
                    help='Number of iterations')
parser.add_argument('--n_bat', type=int, default=100,
                    help='Batch size.')
args = parser.parse_args()


data = get_data_matrix(l_patch=args.l_patch)
d_eff = get_effective_dimensionality(data)

alpha_ = [0.1, 0.15, 0.2]
over_comp_ = [1., 2., 3.]

for alpha, over_comp in product(alpha_, over_comp_):
    n_sp = int(over_comp * d_eff)
    sc = SparseCoder(data=data, n_sp=n_sp, alpha=alpha, n_bat=args.n_bat,
                     d_scale=np.std(data).astype('float32'))

    n_itr = args.n_itr
    scheme = [(n_itr, 0.05), (n_itr, 0.02), (n_itr, 0.01)]
    for n_itr, eta in scheme:
        cost_list=[]
        i_idx = sc.train(n_itr=n_itr, eta=eta, cost_list=cost_list, n_g_itr=100)

    path = os.path.join(
        args.output_dir,
        'sparse_coder_alpha_{:.2f}_overcomp_{:0.2f}.pkl'.format(
            alpha, over_comp))
    print 'Saving to {}'.format(path)
    sc.save(path)
