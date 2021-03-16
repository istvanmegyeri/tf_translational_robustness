from argparse import ArgumentParser
import numpy as np
from scipy import io
import util
import time


# Reason for conversion
# loading test set in mat format 5.1 s, 283 MB
# loading test set in npz format 3.9 s, 157 MB

def main(params):
    print(params)
    if params.in_fname.endswith('.mat'):
        testmat = io.loadmat(params.in_fname)
        x = np.array(np.transpose(testmat['testxdata'], axes=(0, 2, 1)), dtype=np.bool)
        y = np.array(testmat['testdata'][:, 125:815], dtype=np.bool)
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
