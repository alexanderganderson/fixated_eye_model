# Runs the algorithm using no prior on the image
#  This is accomplished by making the dictionary the identity
#  and setting the regularization constant to zero

from model import *
from analyzer import *

L_I = 10 # Linear dimension of the image
N_pix = L_I ** 2
D = np.eye(N_pix).astype('float32')

ig = ImageGenerator(L_I)
ig.make_E()
ig.normalize()

S_gen = ig.img

emb = EMBurak(S_gen, D, N_T = 30, LAMBDA = 0., DC = 0.01)
emb.gen_data()
emb.run()

da = DataAnalyzer(emb.data)

# Plot the Estimated Image and Path after the algorithm ran
da.plot_EM_estimate(da.N_itr - 1)
plt.show()
