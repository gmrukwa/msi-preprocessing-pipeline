"""Convolve spectra with GMM model

Arguments:
    path to mz axis
    path to GMM model root
    path to datasets root

"""

from functools import partial
import os
import sys

import numpy as np

from functional import as_arguments_of, broadcast, for_each, pipe, progress_bar

from components.convolve import convolve
from components.utils import subdirectories


def load_components(components_root):
    fname = partial(os.path.join, components_root)
    preserved = np.loadtxt(fname('indices_after_both.txt'), dtype=int)
    mu = np.loadtxt(fname('mu.txt'))[preserved].ravel()
    sig = np.loadtxt(fname('sig.txt'))[preserved].ravel()
    w = np.loadtxt(fname('w.txt'))[preserved].ravel()
    return mu, sig, w


def input_filename(dataset_root):
    return os.path.join(dataset_root, 'normalized.npy')


def result_filename(dataset_root):
    return os.path.join(dataset_root, 'convolved.npy')


def make_convolver(mzs_path, components_root):
    mzs = np.loadtxt(mzs_path)
    mus, sigs, ws = load_components(components_root)
    return partial(convolve, mzs=mzs, mus=mus, sigs=sigs, ws=ws)


if __name__ == '__main__':
    convolve_dataset = pipe(
        input_filename,
        np.load,
        make_convolver(mzs_path=sys.argv[1], components_root=sys.argv[2])
    )
    convolve_and_save = pipe(
        broadcast(result_filename, convolve_dataset),
        as_arguments_of(np.save)
    )
    convolve_all = pipe(
        subdirectories,
        progress_bar('Dataset'),
        for_each(convolve_and_save, lazy=False)
    )
    convolve_all(sys.argv[3])
