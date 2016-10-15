from itertools import product
import numpy as np
from scipy.io import loadmat
from sparse_coder import SparseCoder, get_effective_dimensionality
from prep_field_dataset import get_data_matrix

data = get_data_matrix(l_patch=20)
d_eff = get_effective_dimensionality(data)

alpha_ = [0.1, 0.15, 0.2]
over_comp_ = [1., 2., 3.]

for alpha, over_comp in product(alpha_, over_comp_):
    n_sp = int(over_comp * d_eff)
    sc = SparseCoder(data=data, n_sp=n_sp, alpha=alpha, n_bat=n_bat,
                     d_scale=np.std(IMAGES).astype('float32'))

    n_itr = 3000
    scheme = [(n_itr, 0.05), (n_itr, 0.02), (n_itr, 0.01)]
    for n_itr, eta in scheme:
        cost_list=[]
        i_idx = sc.train(n_itr=n_itr, eta=eta, cost_list=cost_list, n_g_itr=100)

    path = 'output/sparse_coder_alpha_{:.2f}_overcomp_{:0.2f}.pkl'.format(
        alpha, over_comp)
    print 'Saving to {}'.format(path)
    sc.save(path)
