"""Code to prepare Van Hateren Dataset."""

import os
from argparse import ArgumentParser
import numpy as np
import h5py


def whiten_data_fft(data):
    """
    Use 1/f whitening filter.

    Parameters
    ----------
    data : array, shape (n_imgs, l, l)
        Images to whiten.

    Returns
    -------
    data_wht : array, shape (n_imgs, l, l)
       Whitened images
    """
    n_imgs, num_rows, num_cols = data.shape
    assert num_rows == num_cols
    data -= data.mean(axis=(1, 2), keepdims=True)
    dataFT = np.fft.fftshift(
        np.fft.fft2(data, axes=(1, 2)),
        axes=(1, 2))

    # Build filter
    nyq = np.int32(np.floor(num_rows/2))
    freqs = np.linspace(-nyq, nyq-1, num=num_rows)
    fx, fy = np.meshgrid(freqs, freqs)
    rho = np.sqrt(fx ** 2 + fy ** 2)
    filtf = rho * np.exp(-0.5 * (rho / (0.7 * nyq)) ** 2)

    dataFT_wht = dataFT * filtf[None, :]
    data_wht = np.real(np.fft.ifft2(np.fft.ifftshift(dataFT_wht,
                                          axes=(1, 2)),
                         axes=(1, 2)))
    return data_wht



def _extract_patches(data, n_patches, l_patch, std_thresh=None):
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
    std_thresh : float
        Only choose patches with standard deviation greater than this.

    Returns
    -------
    new_data : array, shape (n_patches, l_patch ** 2)
        Array of patches.
    """
    if std_thresh is None:
        std_thresh = np.std(data[0:50]) * 0.5
    n_imgs, l_i, _ = data.shape
    new_data = np.zeros((n_patches, l_patch ** 2), dtype='float32')
    i = 0
    while i < n_patches:
        q = np.random.randint(n_imgs)
        u, v = np.random.randint(l_i-l_patch, size=2)
        datum = data[q, u:u+l_patch, v:v+l_patch].ravel()
        if datum.std() > std_thresh:
            new_data[i] = datum
            i = i + 1
    return new_data


parser = ArgumentParser('load and whiten natural images')
parser.add_argument('--raw_img_file', type=str,
                    default='/home/redwood/data/vanhateren/images_curated.h5',
                    help='File to load data')
parser.add_argument('--whiten_images', default=False, action='store_true',
                    help='Load images and whiten them.')
parser.add_argument('--whit_img_file', type=str,
                    default='data/final/whitened_images.h5',
                    help='File to save / load whitened images')

parser.add_argument('--extract_patches', default=False, action='store_true',
                    help='Extract patches')
parser.add_argument('--l_patch', type=int, default=32,
                    help='Size of patches to extract')
parser.add_argument('--num_patches', type=int, default=1000,
                    help='Number of patches to extract')
parser.add_argument('--patch_file', type=str,
                    default='data/final/extracted_patches.h5')
args = parser.parse_args()


if args.whiten_images:
    key = 'van_hateren_good' # this key has the good images
    with h5py.File(args.raw_img_file, 'r') as f:
        images = f[key].value

    whit_images = np.zeros_like(images, dtype='float32')

    for i in range(images.shape[0]):
        if i % 10 == 0:
            print('Fourier Transforming Image {}'.format(i))
        whit_images[i] = whiten_data(images[i:i+1])[0]

    with h5py.File(args.whit_img_file, 'w') as f:
        f.create_dataset('images', data=whit_images)


if args.extract_patches:
    with h5py.File(args.whit_img_file, 'r') as f:
        whit_images = f['images']
        patches = _extract_patches(
            whit_images, args.num_patches, args.l_patch)
    with h5py.File(args.patch_file, 'w') as f:
        f['patches'] = patches
        f['l_patch'] = args.l_patch
    print('Patches saved to {}'.format(args.patch_file))

