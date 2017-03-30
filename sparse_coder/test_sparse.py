import os
from argparse import ArgumentParser
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

data = get_data_matrix(l_patch=args.l_patch)

def test_d_eff():
    data = np.random.randn(100, 10)
    d = get_effective_dimensionality(data)

def test_save_restore():
    data = np.random.randn(10, 16 ** 2)
    n_sp = 10
    alpha = 0.1
    n_bat = 2

    sc = SparseCoder(data=data, n_sp=n_sp, alpha=alpha, n_bat=n_bat,
                     d_scale=np.std(data).astype('float32'), sparsify=True)

    n_itr = 1
    eta = 0.01
    cost_list=[]
    i_idx = sc.train(n_itr=n_itr, eta=eta, cost_list=[], n_g_itr=1)

    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    path = os.path.join('tmp', 'test.pkl')
    sc.save(path)

    sc1 = SparseCoder.restore(path=path, data=data)

