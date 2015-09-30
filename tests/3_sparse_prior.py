"""
Code to compare different prior strengths
"""

import numpy as np
from scipy.io import loadmat

from src.model import EMBurak
from utils.image_gen import ImageGenerator

# Tests the algorithm using a '0' from mnist and a sparse coding dictionary
try:
    data = loadmat('sparse_coder/output/mnist_dictionary_pos.mat')
    D = data['D']
except IOError:
    print 'Need to have a dictionary file'
    raise IOError

_, N_pix = D.shape
L_I = int(np.sqrt(N_pix))  # Linear dimension of image

ig = ImageGenerator(L_I)
ig.make_digit()
ig.normalize()

output_dir = 'sparsity'

S_gen = ig.img
S_gen_name = ig.img_name


for LAMBDA in [0.0, 0.01, 0.1, 0.5]:
    emb = EMBurak(S_gen, D, n_t=100, save_mode=True,
                  s_gen_name=S_gen_name, dc_gen=100., dc_infer=100.,
                  output_dir=output_dir, LAMBDA=LAMBDA)
    emb.gen_data()
    emb.run_EM()

    emb.save()

# convert -set delay 30 -colorspace GRAY -colors 256 -dispose 1 -loop 0 -scale 50% *.png alg_performance.gif

# convert -set delay 30 -colors 256 -dispose 1 -loop 0 *.jpg alg_performance.gif
