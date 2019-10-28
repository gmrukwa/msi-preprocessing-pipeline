"""Reduce dataset to selected peaks only

Arguments:
    peak list path
    smoothing window size (int)
    datasets root

"""
from functools import partial
import os
import sys

from functional import for_each, pipe, progress_bar
import numpy as np

import spectrum.smoothing
import spectrum.peak
from utils import subdirectories


def input_filename(dataset_root: str) -> str:
    return os.path.join(dataset_root, 'normalized.npy')


def result_filename(dataset_root: str) -> str:
    return os.path.join(dataset_root, 'selected_peaks.npy')


def select(array, indices):
    return array[indices]


if __name__ == '__main__':
    peak_list = np.loadtxt(sys.argv[1], dtype=int)
    smoothen = partial(spectrum.smoothing.savitzky_golay,
                       window=int(sys.argv[2]))

    smoothen_and_pick = pipe(
        smoothen,
        partial(select, indices=peak_list),
        partial(np.ndarray.astype, dtype=np.float32)
    )

    for dataset_root in progress_bar('dataset')(subdirectories(sys.argv[3])):
        dataset = np.load(input_filename(dataset_root))
        dataset = progress_bar('spectrum')(dataset)
        np.save(result_filename(dataset_root),
                for_each(smoothen_and_pick, parallel=True)(dataset))
