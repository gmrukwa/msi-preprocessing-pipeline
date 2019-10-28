import numpy as np
from numba import jit, prange


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
