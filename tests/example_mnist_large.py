from model import *
from analyzer import *

# Tests the algorithm using a variety of mnist digits and a sparse coding dictionary

N_diff_images = 1 # Run for <> different images
N_replicates = 2 # Run each image <> times

try:
    data = loadmat('data/mnist_dictionary.mat')
    D = data['D']
except IOError:
    print 'Need to have a dictionary file'
    raise IOError

_, N_pix = D.shape
L_I = int(np.sqrt(N_pix)) # Linear dimension of image

ig = ImageGenerator(L_I)

for _ in range(N_diff_images):
    ig.make_digit(mode = 'random')
    ig.normalize()
    S_gen = ig.img
    emb = EMBurak(S_gen, D, N_T = 20, save_mode = True)
    for _ in range(N_replicates):
        emb.reset()
        emb.gen_data()
        emb.run_EM(N_itr = 5)
        filename = emb.save()
        
