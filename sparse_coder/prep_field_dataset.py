"""Code to load in Field Dataset."""

import os
import numpy as np

def get_data_matrix(path=None, l_patch=20):
    """
    Return a data matrix of randomly sampled patches.

    Parameters
    ----------
    path : str
        Path to IMAGES.npy
    l_patch :
        Size of patches to extract.

    Returns
    -------
    data : array, shape (n_patches, l_patch ** 2)
        Random patches from field dataset.
    """
    if path is None:
        data_dir = "data/final/"
        filename = 'IMAGES.npy'
        path = os.path.join(data_dir, filename)
    IMAGES = np.load(path)

    n_patches = int(IMAGES.size / l_patch ** 2 * 10)

    std_thresh = IMAGES.std() * 0.5

    data = extract_patches(IMAGES, n_patches, l_patch, std_thresh)
    return data


def _extract_patches(IMAGES, n_patches, l_patch, std_thresh):
    """
    Extract patches from a set of images.

    Parameters
    ----------
    IMAGES : array, shape (height, width, n_imgs)
        Original images.
    n_patches : int
        Number of patches to extract.
    l_patch : int
        Length and width of patches to extract.
    std_thresh : float
        Only choose patches with standard deviation greater than this.

    Returns
    -------
    data : array, shape (n_patches, l_patch ** 2)
        Array of patches.
    """
    l_i, _, n_imgs = IMAGES.shape
    data = np.zeros((n_patches, l_patch ** 2), dtype='float32')
    i = 0
    while i < n_patches:
        q = np.random.randint(n_imgs)
        u, v = np.random.randint(l_i-l_patch, size=2)
        datum = IMAGES[u:u+l_patch, v:v+l_patch, q].ravel()
        if datum.std() > std_thresh:
            data[i] = datum
            i = i + 1
    return data

