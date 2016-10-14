"""
Test hyperacuity abilities of the lattice.
"""

import numpy as np
from argparse import ArgumentParser
from itertools import product

from src.model import EMBurak
from utils.image_gen import ImageGenerator


parser = ArgumentParser('Test the motion benefit')
parser.add_argument('--n_repeats', type=int, default=2,
                    help='Number of repetitions for each set of parameters.')
parser.add_argument('--n_t', type=int, default=500,
                    help='Number of timesteps')
parser.add_argument('--n_g_itr', type=int, default=160,
                    help='Number of gradient steps per M iteration')
parser.add_argument('--output_dir', type=str, default='hyper_acuity',
                    help='Output_directory')
#  parser.add_argument('--dc', type=float, default=20.,
#                      help='Diffusion Constant for eye motion')
#  parser.add_argument('--image', type=str, default='e',
#                      help='Digit type')
#  parser.add_argument('--ds', type=float, default=0.4,
#                      help='Pixel Spacing')

# TODO:
# Sweep contrast
args = parser.parse_args()
n_itr = args.n_t
de = 1.  # Neuron spacing
L_I = 27  # Image size
fs = 1 / de  # Sampling
fn = 0.5 * fs  # Max frequency for sampling lattice - Nyquist
f_gabor = 0.3  # Frequency for generating the gabor

# FIXME: perhaps we can use a better dictionary here
D = np.eye(L_I ** 2)

motion_info_ = [
    ({'mode': 'Experiment', 'fpath': 'data/paths.mat'},
     {'mode': 'PositionDiffusion', 'dc': 20.}),
    ({'mode': 'Diffusion', 'dc': 0.5},
     {'mode': 'PositionDiffusion', 'dc': 0.5})
][1:2]

f_ = [fn * r for r in [1.25, 1, 1.5]]

for (motion_gen, motion_prior), f in product(motion_info_, f_):
    ig = ImageGenerator(L_I)
    ds = f_gabor / f

    emb = EMBurak(
        ig.img, D, motion_gen, motion_prior, n_t=args.n_t, save_mode=True,
        s_gen_name=ig.img_name, ds=ds, neuron_layout='hex', fista_c=0.8,
        de=de, l_n=5, n_itr=n_itr, lamb=0.0, n_g_itr=args.n_g_itr, l1=200.,
        output_dir_base=args.output_dir, s_range='sym', print_mode=False)
    for u in range(args.n_repeats):
        phi = np.random.rand(1)[0] * np.pi * 2
        eta = np.pi * ((u % 2) * 2 + 1) / 4.
        ig.make_gabor(f=f_gabor, sig=L_I/3., phi=phi, eta=eta)

        XR, YR, R = emb.gen_data(ig.img - ig.img.mean())
        emb.run_em(R)
        emb.save()
        emb.reset()

