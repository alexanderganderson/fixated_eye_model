import numpy as np
import matplotlib.pyplot as plt

from src.model import EMBurak
from src.analyzer import DataAnalyzer
from utils.image_gen import ImageGenerator
from utils.image_gen import rotated_e
# Simulates the rotating E experiment

D = rotated_e().reshape((4, 16))

_, N_pix = D.shape

L_I = int(np.sqrt(N_pix))  # Linear dimension of image

ig = ImageGenerator(L_I)
# ig.make_digit(mode = 'random')
ig.make_digit()
ig.normalize()

q = 0
s_gen = D[q]
s_gen_name = {0: 'E_down', 1: 'E_up', 2: 'E_right', 3: 'E_left'}[q]

motion_gen = {'mode': 'Experiment', 'fpath': 'data/resampled_data.mat'}
motion_prior = {'mode': 'PositionDiffusion', 'dc': 100.}
# motion_prior = {'mode': 'VelocityDiffusion',
#                 'v0': np.sqrt(40.), 'dcv': 50.}

emb = EMBurak(s_gen, D, motion_gen, motion_prior, n_t=50, save_mode=True,
              s_gen_name=s_gen_name, n_itr=10, lamb=0.0)
XR, YR, R = emb.gen_data(s_gen)
emb.run_EM(R)

emb.save()

# da = DataAnalyzer(emb.data)

# # Plot the Estimated Image and Path after the algorithm ran
# da.plot_EM_estimate(da.N_itr - 1)
# plt.show()
