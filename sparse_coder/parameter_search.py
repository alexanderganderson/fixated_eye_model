from itertools import product
import numpy as np
from scipy.io import loadmat
from scipy.linalg import eigh
from sparse_coder import SparseCoder

data_dir = "data/final/"

IMAGES = np.load(data_dir + 'IMAGES.npy')



l_patch = 20
n_bat = 200
n_patches = int(IMAGES.size / l_patch ** 2 * 10)

# Build data matrix
data = np.zeros((n_patches, l_patch ** 2), dtype='float32')
std_thresh = IMAGES.std() * 0.5
l_i, _, n_imgs = IMAGES.shape
i = 0
while i < n_patches:
    q = np.random.randint(n_imgs)
    u, v = np.random.randint(l_i-l_patch, size=2)
    datum = IMAGES[u:u+l_patch, v:v+l_patch, q].ravel()
    if datum.std() > std_thresh:
        data[i] = datum
        i = i + 1

IM_cov = np.cov(data.T)
evals, evecs = eigh(IM_cov)
cs = np.cumsum(evals[::-1])
d_eff = np.argmax(cs > cs[-1] * 0.9) # Effective Dim of data

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
