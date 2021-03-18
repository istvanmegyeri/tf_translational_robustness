from argparse import ArgumentParser
import numpy as np
from scipy import io
import h5py
import util
import time
import os

# Reason for conversion
# loading test set in mat format 5.1 s, 283 MB
# loading test set in npz format 3.9 s, 157 MB

def main(params):
    print(params)
    if params.in_fname.endswith('.mat'):
        if params.in_fname.endswith('test.mat'):
            mat_file = io.loadmat(params.in_fname)
        else:
            mat_file = h5py.File(params.in_fname, 'r')
        keys=list(filter(lambda k:not k.startswith("_"),mat_file.keys()))
        data_key=list(filter(lambda k:k.endswith('xdata'),keys))[0]
        lab_key=list(filter(lambda k:not k.endswith('xdata'),keys))[0]
        print(data_key,lab_key)
        x = np.array(np.transpose(mat_file[data_key], axes=(2, 0, 1)), dtype=np.bool)
        y = np.array(mat_file[lab_key], dtype=np.bool).T[:, 125:815]
    else:
        raise Exception('Unsupported file format: {0}'.format(params.in_fname))

    if params.out_fname.endswith('npz'):
        util.mk_parent_dir(params.out_fname)
        np.savez_compressed(params.out_fname, x=x, y=y)
    else:
        raise Exception('Unsupported output format:{0}'.format(params.out_fname))


if __name__ == '__main__':
    parser = ArgumentParser(description='App description')
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--out_fname', type=str, required=True)
    parser.add_argument('--in_fname', type=str, required=True)
    FLAGS = parser.parse_args()
    np.random.seed(9)
    main(FLAGS)
