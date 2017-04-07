"""Code to prepare Van Hateren Dataset."""

import os
from argparse import ArgumentParser
import numpy as np
import h5py

filter_store = {}

def _build_whitening_filter(n):
    if n in filter_store:
        return filter_store[n]

    nyq = np.int32(np.floor(n/2))
    freqs = np.linspace(-nyq, nyq-1, num=n)
    fx, fy = np.meshgrid(freqs, freqs)
    rho = np.sqrt(fx ** 2 + fy ** 2)
    filtf = rho * np.exp(-0.5 * (rho / (0.7 * nyq)) ** 2)
    filter_store[n] = filtf
    return filtf

def _whiten_patch_fft(data):
    """
    Use 1/f whitening filter.

    Parameters
    ----------
    data : array, shape (l, l)
        Images to whiten.

    Returns
    -------
    data_wht : array, shape (l, l)
       Whitened images
    """
    num_rows, num_cols = data.shape
    assert num_rows == num_cols
    #  data -= data.mean(axis=0, keepdims=True)
    dataFT = np.fft.fftshift(np.fft.fft2(data))

    filtf = _build_whitening_filter(num_rows)

    dataFT_wht = dataFT * filtf
    data_wht = np.real(np.fft.ifft2(np.fft.ifftshift(dataFT_wht)))
    return data_wht

def whiten_patches_fft(data, in_place=False):
    if in_place:
        whitened_patches = data
    else:
        whitened_patches = np.zeros_like(data)

    for i in range(data.shape[0]):
        whitened_patches[i] = _whiten_patch_fft(data[i])
    return whitened_patches


def _extract_patches(data, n_patches, l_patch, criterion):
    """
    Extract patches from a set of images.

    Parameters
    ----------
    data: array, shape (n_imgs, height, width)
        Original images.
    n_patches : int
        Number of patches to extract.
    l_patch : int
        Length and width of patches to extract.
    criterion : func
        Function that takes in a patch and outputs True of False
            to determine if we should use that patch.

    Returns
    -------
    new_data : array, shape (n_patches, l_patch, l_patch)
        Array of patches.
    """
    n_imgs, l_i, _ = data.shape
    new_data = np.zeros((n_patches, l_patch, l_patch), dtype='float32')
    i = 0
    ii = 0
    while i < n_patches:
        ii += 1
        q = np.random.randint(n_imgs)
        u, v = np.random.randint(l_i-l_patch, size=2)
        datum = data[q, u:u+l_patch, v:v+l_patch]
        if criterion(datum):
            new_data[i] = datum
            i = i + 1
            if i % 10000 == 0:
                print('Extracting patch {}'.format(i))
    print 'Used {:0.2f} percent of patches'.format((100. * i) / ii)
    return new_data


parser = ArgumentParser('load and whiten natural images')
parser.add_argument('--raw_img_file', type=str,
                    default='/home/redwood/data/vanhateren/images_curated.h5',
                    help='File to load data')
#  parser.add_argument('--whiten_images', default=False, action='store_true',
#                      help='Load images and whiten them.')
#  parser.add_argument('--whit_img_file', type=str,
#                      default='data/final/whitened_images.h5',
#                      help='File to save / load whitened images')

parser.add_argument('--extract_patches', default=False, action='store_true',
                    help='Extract patches')
parser.add_argument('--l_patch', type=int, default=32,
                    help='Size of patches to extract')
parser.add_argument('--num_patches', type=int, default=1000 * 500,
                    help='Number of patches to extract')
parser.add_argument('--patch_file', type=str,
                    default='data/final/new_extracted_patches.h5')
args = parser.parse_args()


if args.extract_patches:
    key = 'van_hateren_good' # this key has the good images
    with h5py.File(args.raw_img_file, 'r') as f:
        images = f[key].value
        #  images = f[key]

    criterion = lambda x: True
    n_patches_test = int(args.num_patches / 50)
    patches = _extract_patches(
            images, n_patches_test, args.l_patch, criterion)
    patch_mean = patches.mean(axis=0, keepdims=True)
    patches -= patch_mean

    white_patches = whiten_patches_fft(patches)
    std = white_patches.std()

    criterion = lambda x: _whiten_patch_fft(x - patch_mean[0]).std() > std
    patches = _extract_patches(
        images, args.num_patches, args.l_patch, criterion)
    patches -= patch_mean

    white_patches = whiten_patches_fft(patches)



    with h5py.File(args.patch_file, 'w') as f:
        #  f['patches'] = patches
        f['white_patches'] = white_patches
        f['l_patch'] = args.l_patch
    print('Patches saved to {}'.format(args.patch_file))




