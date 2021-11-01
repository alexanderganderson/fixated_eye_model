"""Code to save and load dictionaries from a h5py file."""

import os
import numpy as np
import h5py


def save_dict(fn, d):
    """
    Save a dictionary using h5py.

    Parameters
    ----------
    fn: str
        File to save the dictionary
    d : dict
        Dictionary

    Assumes all keys are strings
    Assumes values are [str, int, float, np.ndarray, dict]
    """
    with h5py.File(fn, 'w') as f:
        _save_dict(f, d)


def _save_dict(f, d, base=''):
    """
    Recursive helper function for saving dictionaries.

    Parameters
    ----------
    f : h5py.File
        Writable file to save objects
    d : dict
        Dictionary [see save_dict for assumptions]
    base : str
        Base string to save files
    """
    for k, v in d.iteritems():
        k1 = os.path.join(base, str(k))
        if isinstance(v, (str, np.ndarray, int, float)):
            f[k1] = v
        elif isinstance(v, dict):
            _save_dict(f, v, base=k1)
        else:
            raise ValueError('Unknown type for v: {}'.format(type(v)))


def get_dict(fn, base=''):
    """
    Return a dictionary from the file starting at base.
    """
    with h5py.File(fn, 'r') as f:
        return _get_dict(f, base)


def _get_dict(f, base):
    if base == '':
        u = f
    else:
        u = f[base]
    d = {}
    if isinstance(u, h5py.Dataset):
        return u.value
    for k, v in u.iteritems():
        if isinstance(v, h5py.Dataset):
            d[k] = v.value
        elif isinstance(v, h5py.Group):
            d[k] = _get_dict(f, os.path.join(base, k))
        else:
            raise ValueError('Unknown type for v: {}'.format(type(v)))
    return d


class LazyDict(object):
    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, val):
        return get_dict(self.fn, val)

    def __setitem__(self, key, val):
        with h5py.File(self.fn, 'r+') as f:
            f[key] = val


if __name__ == '__main__':
    d = {
        'a': 'a',
        'b': 5,
        'c': np.zeros(5),
        'd': {
            '0': {
                'e': np.zeros(5),
                'f': 4
            },
            '1': {
                'e': np.zeros(5),
                'f': 5
            },
            'g': 'asdf'
        }
    }
    fn = 'test.h5'
    save_dict(fn, d)
    d1 = get_dict(fn)

    print d
    print d1
    os.remove(fn)

