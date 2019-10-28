from typing import Tuple

import numpy as np


class Spectrum:
    def __init__(self, mz: np.ndarray, counts: np.ndarray):
        if mz.size != counts.size:
            raise ValueError('Expected mz and count to be equal size. Was '
                             f'{mz.size} and {counts.size}')
        self.mz = mz
        self.counts = counts


def gradient_detection(spectrum: Spectrum) -> Tuple[np.ndarray, Spectrum]:
    first_differential = np.gradient(spectrum.counts)
    second_differential = np.gradient(first_differential)
    local_extrema = np.nonzero(
        first_differential[:-1] * first_differential[1:] <= 0)
    potential_maxima = np.nonzero(second_differential[local_extrema] < 0)[0]
    right_potential_maxima = np.clip(potential_maxima + 1, a_min=0, a_max=local_extrema[0].size-1)
    left_potential_maxima = np.clip(potential_maxima - 1, a_min=0, a_max=local_extrema[0].size-1)
    counts_at_potential_maximas = np.vstack([
        spectrum.counts[(local_extrema[0][(left_potential_maxima,)],)],
        spectrum.counts[(local_extrema[0][(potential_maxima,)],)],
        spectrum.counts[(local_extrema[0][(right_potential_maxima,)],)]
    ])
    true_maxima = np.argmax(counts_at_potential_maximas, axis=0)
    indices = np.unique(np.sort(np.concatenate([
        local_extrema[0][left_potential_maxima[true_maxima == 0]],
        local_extrema[0][potential_maxima[true_maxima == 1]],
        local_extrema[0][right_potential_maxima[true_maxima == 2]]
    ])))
    return indices, Spectrum(spectrum.mz[(indices,)], spectrum.counts[(indices,)])
