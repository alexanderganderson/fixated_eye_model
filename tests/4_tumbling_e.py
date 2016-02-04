import numpy as np
import matplotlib.pyplot as plt

from src.model import EMBurak
from utils.image_gen import rotated_e
# Simulates the rotating E experiment


def get_probs(emb, XR, YR, R, imgs):
    n_imgs = imgs.shape[0]
    p = np.zeros((n_imgs,))
    for i in range(n_imgs):
        p[i] = emb.ideal_observer_cost(XR, YR, R, imgs[i])
    p -= p.mean()
    p = np.exp(-p)
    p = p / p.sum()
    return p


def get_many_probs(emb, n_runs=100):
    p_ = np.zeros((n_runs, 4))
    for u in range(n_runs):
        XR, YR, R = emb.gen_data(s_gen, print_mode=False)
        p_[u] = get_probs(emb, XR, YR, R, imgs)
    return p_

n_runs = 10
imgs = rotated_e()
D = imgs.reshape((4, 25))

q = 2
s_gen = imgs[q]
s_gen_name = {0: 'E_down', 1: 'E_up', 2: 'E_right', 3: 'E_left'}[q]

motion_prior = {'mode': 'PositionDiffusion', 'dc': 20.}
# motion_prior = {'mode': 'VelocityDiffusion',
#                 'v0': np.sqrt(40.), 'dcv': 50.}


# No eye motion
motion_gen = {'mode': 'Diffusion', 'dc': 0.00001}
emb = EMBurak(s_gen, D, motion_gen, motion_prior, n_t=100, save_mode=True,
              s_gen_name=s_gen_name, ds=0.57, neuron_layout='hex',
              de=1.09, l_n=6,
              n_itr=10, lamb=0.0, tau=0.05)

p1_ = get_many_probs(emb, n_runs=n_runs)

# Real motion
motion_gen = {'mode': 'Experiment', 'fpath': 'data/paths.mat'}
emb = EMBurak(s_gen, D, motion_gen, motion_prior, n_t=100, save_mode=True,
              s_gen_name=s_gen_name, ds=0.57, neuron_layout='hex',
              de=1.09, l_n=6,
              n_itr=10, lamb=0.0, tau=0.05)

p2_ = get_many_probs(emb, n_runs=n_runs)

plt.hist(p1_[q], label='No Motion', n_bins=50)
plt.hist(p2_[q], label='True Eye Motion', n_bins=50)
plt.legend()
plt.show()
