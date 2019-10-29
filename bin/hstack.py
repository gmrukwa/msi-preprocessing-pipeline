import argparse
import os

import numpy as np
import pandas as pd

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--dst', help='destination csv', required=True)
    args.add_argument('--src', nargs='+', help='source files to be concatenated')
    return args.parse_args()


def load_csv(fname):
    return pd.read_csv(fname)


def load_npy(fname):
    data = np.load(fname)
    _, name = os.path.split(fname)
    name, _ = os.path.splitext(name)
    if data.ndim == 1 or data.shape[1] == 1:
        names = [name]
    else:
        names = ['{0}_{1}'.format(name, i) for i in range(data.shape[1])]
    return pd.DataFrame(data=data, columns=names)


def load_file(fname):
    _, ext = os.path.splitext(fname)
    if ext.endswith('csv'):
        return load_csv(fname)
    if ext.endswith('npy'):
        return load_npy(fname)
    raise ValueError('unsupported extension ({0}) of {1}'.format(ext, fname))


if __name__ == '__main__':
    args = parse_args()
    data = [load_file(f) for f in args.src]
    joint = pd.concat(data, axis=1)
    joint.to_csv(args.dst, index=None)
