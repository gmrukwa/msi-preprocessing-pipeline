# -*- coding: utf-8 -*-
"""
Created on Tue May 22 18:36:28 2018

Extract mean spectrum from datasets.

Arguments:
    - destination path of mean spectrum
    - paths datasets root
    - dataset file name (inside subdirectories)
    - (optional) binary outlier filter file name

@author: Grzegorz Mrukwa
"""

from functools import partial
import os
import sys

import numpy as np

from functional import as_arguments_of, broadcast, for_each, pipe, progress_bar
from components.utils import subdirectories


def input_filename(dataset_root):
    return os.path.join(dataset_root, sys.argv[3])


def conditionally_filtered_dataset(dataset_root):
    spectra = np.load(input_filename(dataset_root))
    if len(sys.argv) > 4:
        filter_path = os.path.join(dataset_root, sys.argv[4])
        preserved = np.logical_not(np.load(filter_path))
        spectra = spectra[preserved]
    return spectra


process_dataset = pipe(
    conditionally_filtered_dataset,
    broadcast(
        partial(np.mean, axis=0),
        partial(np.size, axis=0)
    )
)


def average(data, weights):
    return np.average(data, axis=0, weights=weights)


def transpose(collection):
    return zip(*collection)


process_all_datasets = pipe(
    subdirectories,
    progress_bar('dataset'),
    for_each(process_dataset, lazy=False),
    transpose,
    as_arguments_of(average),
    partial(np.savetxt, sys.argv[1])
)


if __name__ == '__main__':
    process_all_datasets(sys.argv[2])
