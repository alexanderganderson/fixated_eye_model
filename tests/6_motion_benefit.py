"""
Script to show the benefit of eye motions.

(1) Spikes generated by no motion.
(2) Spikes generated including motion.

See which case it is easier to reconstruct the image.

Uses EM decoder and No Motion Decoder
"""

import numpy as np
from scipy.io import loadmat
from argparse import ArgumentParser

from itertools import product

from src.model import EMBurak
from utils.image_gen import ImageGenerator

parser = ArgumentParser('Test the motion benefit')
parser.add_argument('--n_repeats', type=int, default=20,
                    help='Number of repetitions for each set of parameters.')
parser.add_argument('--n_t', type=int, default=700,
                    help='Number of timesteps')
parser.add_argument('--output_dir', type=str, default='motion_benefit_test',
                    help='Output_directory')
parser.add_argument('--image', type=str, default='e',
                    help='Digit type')
parser.add_argument('--experiment', type=str, default='motion_benefit',
                    help='Tag to determine type of experiment to run')


args = parser.parse_args()

n_itr = args.n_t / 5

if args.image == 'e':
    L_I = 20
    ig = ImageGenerator(L_I)
    ig.make_big_e()
    ig.normalize()
    D = np.eye(L_I ** 2)
    from utils.block_prior import block_prior
    D = 0.25 * block_prior(L_I / 2)
#  elif args.image == 'mnist':
#      # Sparse coding dictionary prior
#      data = loadmat('sparse_coder/output/mnist_dictionary.mat')
#      D = data['D']
#      _, N_pix = D.shape
#      L_I = int(np.sqrt(N_pix))  # Linear dimension of image
#      ig = ImageGenerator(L_I)
#      ig.make_digit()
#      ig.normalize()
else:
    raise ValueError('Unrecognized image: {}'.format(args.image))

if args.experiment == 'motion_benefit':
    # Main experiment concerning motion benefit using motion and no motion
    motion_info_ = [
        ({'mode': 'Experiment', 'fpath': 'data/paths.mat'},
         {'mode': 'PositionDiffusion', 'dc': 20.}),
        ({'mode': 'Diffusion', 'dc': 0.0001},
         {'mode': 'PositionDiffusion', 'dc': 20.}),
    ]
    drop_prob = None
    run_em = True
    run_nom = True
elif args.experiment == 'motion_benefit_drop':
    # Motion benefit after dropping out cones
    motion_info_ = [
        ({'mode': 'Experiment', 'fpath': 'data/paths.mat'},
         {'mode': 'PositionDiffusion', 'dc': 20.}),
        ({'mode': 'Diffusion', 'dc': 0.0001},
         {'mode': 'PositionDiffusion', 'dc': 20.}),
    ]
    drop_prob_ = [0.3]
    run_em = True
    run_nom = False
elif args.experiment == 'no_motion_best_dc_infer':
    # Sweep over diffusion constants to find best amount of motion for each
    # amount of cone drop out
    motion_info_ = [
        ({'mode': 'Diffusion', 'dc': 0.001},
         {'mode': 'PositionDiffusion', 'dc': dc_infer})
        for dc_infer in [0.01, 0.4, 2., 20., 100.]]
    drop_prob_ = [None]
    run_em = True
    run_nom = False
elif args.experiment == 'best_dc_cone_drop':
    motion_info_ = [
        ({'mode': 'Diffusion', 'dc': dc_gen},
         {'mode': 'PositionDiffusion', 'dc': dc_infer})
        for dc_gen, dc_infer in [
            #  [0.01, 0.01],
            #  [0.01,   2.],
            #  [0.01,  10.],
            #  [0.4, 0.4],
            #  [0.4, 4.],
            #  [0.4, 20.],
            [2., 2.],
            [2., 10.],
            [2., 20,],
            [4., 2.],
            [4., 10.],
            [4., 20.],
            [8., 8.],
            [8., 20.],
            #  [20., 20.],
            #  [40., 40.],
            #  [100., 100.],
        ]
    ]

    drop_prob_ = [0.0, 0.3, 0.5]
    run_em = True
    run_nom = False
else:
    raise ValueError('Invalid experiment name')

ds_ = [0.40]
de = 1.09

for (motion_gen, motion_prior), ds, drop_prob in product(
    motion_info_, ds_, drop_prob_):
    emb = EMBurak(
        l_i=L_I,
        d=D,
        motion_gen=motion_gen,
        motion_prior=motion_prior,
        n_t=args.n_t,
        save_mode=True,
        s_gen_name=ig.img_name,
        ds=ds,
        neuron_layout='hex',
        drop_prob=drop_prob,
        de=de,
        l_n=8.1,
        n_itr=n_itr,
        n_g_itr=None,
        output_dir_base=args.output_dir,
    )

    if run_em:
        emb.n_g_itr = 320
        for _ in range(args.n_repeats):
            XR, YR, R = emb.gen_data(ig.img)
            emb.run_em(R)
            emb.save()
            emb.reset()

    if run_nom:
        emb.n_g_itr = 100
        for _ in range(args.n_repeats):
            XR, YR, R = emb.gen_data(ig.img)
            emb.run_inference_no_motion(R)
            emb.save()
            emb.reset()

    #  for _ in range(args.n_repeats):
    #      XR, YR, R = emb.gen_data(ig.img)
    #      emb.run_inference_true_path(R, XR, YR)
    #      emb.save()
    #      emb.reset()


# convert -set delay 30 -colorspace GRAY
# -colors 256 -dispose 1 -loop 0 -scale 50% *.png alg_performance.gif

# convert -set delay 30 -colors 256
# -dispose 1 -loop 0 *.jpg alg_performance.gif
