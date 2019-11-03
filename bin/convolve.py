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
from numba import jit, prange

from functional import as_arguments_of, broadcast, for_each, pipe, progress_bar

from components.utils import subdirectories


@jit(nopython=True)
def relevant_range(mzs, mu, sig, multiplier=15):
    lower_bound = mu - multiplier * sig
    upper_boud = mu + multiplier * sig
    relevant = np.logical_and(lower_bound <= mzs, mzs <= upper_boud)
    relevant_indices = np.nonzero(relevant)
    return relevant_indices[0][0], relevant_indices[0][-1]


@jit(nopython=True, parallel=True)
def convolve(spectra, mzs, mus, sigs, ws) -> np.ndarray:
    convolved = np.zeros(shape=(mus.size, spectra.shape[0]), dtype=np.float32)
    for i in range(mus.size):
        mu, sig, w = mus[i], sigs[i], ws[i]
        first, last = relevant_range(mzs, mu, sig)
        # normal_distribution - manually for the sake of JIT
        x = (mzs[first:last] - mu) / sig
        y = (1. / (np.sqrt(2 * np.pi) * sig)) * np.exp(-x * x / 2.)
        # weighting components
        component = w * y
        for row in prange(spectra.shape[0]):
            convolved[i, row] = np.sum(spectra[row, first:last] * component)
    return convolved.T


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
