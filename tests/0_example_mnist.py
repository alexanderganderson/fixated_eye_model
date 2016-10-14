"""Basic test of the code."""
import numpy as np
# import matplotlib.pyplot as plt
from scipy.io import loadmat

from src.model import EMBurak
from src.analyzer import DataAnalyzer
from utils.image_gen import ImageGenerator

run_analyzer = False
check_gradient = True


# Tests the algorithm using a '0' from mnist and a sparse coding dictionary

data = loadmat('sparse_coder/output/mnist_dictionary.mat')
D = data['D']

_, N_pix = D.shape

L_I = int(np.sqrt(N_pix))  # Linear dimension of image

ig = ImageGenerator(L_I)
# ig.make_digit(mode = 'random')
ig.make_digit()
ig.normalize()

s_gen = ig.img
s_gen_name = ig.img_name

motion_gen = {'mode': 'Diffusion', 'dc': 100.}
motion_prior = {'mode': 'PositionDiffusion', 'dc': 100.}

emb = EMBurak(s_gen - 0.5, D, motion_gen, motion_prior, n_t=10, save_mode=True,
              s_gen_name=s_gen_name, n_itr=10, lamb=0.0, s_range='sym')
XR, YR, R = emb.gen_data(s_gen)


emb.run_em(R)


    # emb.save()

if run_analyzer:
    da = DataAnalyzer(emb.data)
    da.plot_em_estimate(0)
    print da.snr_list()
    print da.time_list()

    da.plot_image_and_rfs(s=50)

if check_gradient:
    from utils.gradient_checker import hessian_check

    def f(A):
       emb.tc.t_A.set_value(A.astype('float32'))
       return emb.get_spike_cost(R)


    def fpp(A):
       emb.tc.t_A.set_value(A.astype('float32'))
       return emb.get_hessian()

    x0 = emb.tc.get_A()
    #  x0 = np.random.randn(D.shape[0]) * 10
    #  import pdb; pdb.set_trace()
    for _ in range(10):
       u, v = hessian_check(f, fpp, (D.shape[0],), x0=x0)
       print u, v


    #  import pdb; pdb.set_trace()





# # Plot the Estimated Image and Path after the algorithm ran
# da.plot_EM_estimate(da.N_itr - 1)
# plt.show()

# convert -set delay 30 -colorspace GRAY -colors 256
#      -dispose 1 -loop 0 -scale 50% *.png alg_performance.gif

# convert -set delay 30 -colors 256
#     -dispose 1 -loop 0 *.jpg alg_performance.gif
