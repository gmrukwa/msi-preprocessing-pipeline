# -*- coding: utf-8 -*-
"""
Created on Tue May 22 19:15:43 2018

Align each spectrum to mean spectrum.

Arguments:
    path to mz axis
    path to mean spectrum
    path to datasets root

@author: Grzegorz Mrukwa
"""

from functools import partial
import os
import sys

import numpy as np
from tqdm import tqdm

from functional import for_each, pipe, progress_bar
from spectrum.alignment import pafft
from utils import subdirectories


def input_filename(dataset_root):
    return os.path.join(dataset_root, 'lowered.npy')


def result_filename(dataset_root):
    return os.path.join(dataset_root, 'aligned.npy')


if __name__ == '__main__':
    mzs = np.loadtxt(sys.argv[1])
    mean_spectrum = np.loadtxt(sys.argv[2])
    process_spectrum = partial(pafft, mzs=mzs, reference_counts=mean_spectrum)

    align_dataset = pipe(
        input_filename,
        np.load,
        progress_bar('spectrum'),
        for_each(process_spectrum, parallel=True)
    )

    for dataset in tqdm(subdirectories(sys.argv[3]), desc='Dataset'):
        aligned = align_dataset(dataset)
        np.save(result_filename(dataset), aligned)
