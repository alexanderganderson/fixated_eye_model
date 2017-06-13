"""Legacy script for converting pkl files to hdf5 files."""

import os
import pickle as pkl
from context import utils
from utils.h5py_utils import save_dict

output_dir = FIXME

pkl_fns = [os.path.join(output_dir, fn)
           for fn in os.listdir(output_dir)
           if fn.endswith('.pkl')]

print len(pkl_fns)

for pkl_fn in pkl_fns:
    print pkl_fn
    with open(pkl_fn, 'rb') as f:
        data = pkl.load(f)
        save_fn = pkl_fn.replace('pkl', 'h5')
        save_dict(save_fn, data)
