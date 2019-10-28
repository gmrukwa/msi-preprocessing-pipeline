# -*- coding: utf-8 -*-
"""Remove baseline from the dataset with adaptive method.

Created on Sun May 27 21:18:42 2018

Arguments:
    mzs (path-alike) - location of file with mz axis for the dataset
    dataset paths (list of path-alike) - locations of datasets to be processed

Result:
    Saved to the directory with dataset but with 'lowered.npy' name.

@author: Grzegorz Mrukwa
"""

from functools import partial
import os
import sys

from functional import pmap
import numpy as np
from tqdm import tqdm

from components.spectrum.baseline import adaptive_remove


load_mzs = np.loadtxt
load_data = np.load
save_data = np.save


def baseline_remover(mz_axis):
    return partial(adaptive_remove, mz_axis)


def lowered_path(path):
    return os.path.join(os.path.dirname(path), 'lowered.npy')


if __name__ == '__main__':
    mzs = load_mzs(sys.argv[1])
    remove_baseline = baseline_remover(mzs)
    for dataset_path in tqdm(sys.argv[2:], desc='Dataset'):
        spectra = load_data(dataset_path)
        assert spectra.shape[1] == mzs.size, dataset_path
        lowered = pmap(remove_baseline, tqdm(spectra,
                                             desc='Spectra'), chunksize=800)
        result_path = lowered_path(dataset_path)
        save_data(result_path, lowered)
