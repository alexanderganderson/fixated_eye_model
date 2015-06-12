from model import *
from analyzer import *

# Tests the algorithm using a '0' from mnist and a sparse coding dictionary

try:
    data = loadmat('data/mnist_dictionary.mat')
    D = data['D']
except IOError:
    print 'Need to have a dictionary file'
    raise IOError

_, N_pix = D.shape
L_I = int(np.sqrt(N_pix)) # Linear dimension of image

ig = ImageGenerator(L_I)
ig.make_digit()
ig.normalize()

S_gen = ig.img

emb = EMBurak(S_gen, D, N_T = 30)
emb.gen_data()
emb.run()

da = DataAnalyzer(emb.data)

# Plot the Estimated Image and Path after the algorithm ran
da.plot_EM_estimate(da.N_itr - 1)
plt.show()