import os
#os.chdir('../')

from model import *
from analyzer import *


# Tests MNIST

try:
    data = loadmat('data/mnist_dictionary.mat')
    D = data['D']
except IOError:
    print 'Need to have a dictionary file'
    raise IOError

N_L, N_pix = D.shape
L_I = int(np.sqrt(N_pix))

ig = ImageGenerator(L_I)
ig.make_digit()
ig.normalize()

S_gen = ig.img

emb = EMBurak(S_gen, D, N_T = 30)

emb.gen_data()

emb.run()

da = DataAnalyzer(emb.data)
