"""Code to fit velocity prior."""

# import os

from scipy.io import loadmat
from argparse import ArgumentParser

from src.model import EMBurak
from utils.image_gen import ImageGenerator

parser = ArgumentParser('Fit the velocity diffusion prior')
parser.add_argument('--n_repeats', type=int, default=5,
                    help='Number of repetitions for each set of parameters.')
parser.add_argument('--n_t', type=int, default=100,
                    help='Number of timesteps')
parser.add_argument('--output_dir', type=str, default='fit_vdm',
                    help='Output_directory')
parser.add_argument('--dc', nargs='*', type=float, default=[],
                    help='Diffusion Constant for motion prior')
parser.add_argument('--v0', nargs='*', type=float, default=[],
                    help='Initial velocities for VDM')
parser.add_argument('--dcv', nargs='*', type=float, default=[],
                    help='Velocity diffusion constants')
parser.add_argument('--n_p', type=int, default=20,
                    help='Number of particles for particle filter')

args = parser.parse_args()

D = loadmat('sparse_coder/output/mnist_dictionary.mat')['D']
ig = ImageGenerator(14)
ig.make_digit()
ig.normalize()

motion_gen = {'mode': 'Experiment', 'fpath': 'data/paths.mat'}

if len(args.dcv) is not len(args.v0):
    raise ValueError('Must have the same number of dcv and v0 values.')
motion_prior_ = [{'mode': 'VelocityDiffusion', 'v0': v0, 'dcv': dcv}
                 for v0, dcv in zip(args.dcv, args.v0)]
motion_prior_ += [{'mode': 'PositionDiffusion', 'dc': dc} for dc in args.dc]


for motion_prior in motion_prior_:
    emb = EMBurak(
        ig.img, D, motion_gen, motion_prior, n_t=args.n_t, save_mode=True,
        s_gen_name=ig.img_name, ds=0.57, neuron_layout='hex',
        de=1.09, l_n=6, print_mode=False, output_dir_base=args.output_dir,
        n_p=args.n_p)
    for _ in range(args.n_repeats):
        XR, YR, R = emb.gen_data(ig.img)
        emb.run_em(R)
        emb.save()
        emb.reset()
