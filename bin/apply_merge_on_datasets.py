"""Apply found merge on datasets

Arguments:
    path to merged artifacts root
    path to datasets root

"""
from functools import partial
import os
import sys

from functional import pipe, progress_bar
import numpy as np

import components.spectrum.model as mdl
from components.utils import subdirectories


def input_filename(dataset_root):
    return os.path.join(dataset_root, 'convolved.npy')


def result_filename(dataset_root):
    return os.path.join(dataset_root, 'merged.npy')


def load_matches(matches_root) -> mdl.Matches:
    indices = np.loadtxt(
        os.path.join(matches_root, 'merged_start_indices.txt'),
        dtype=int)
    lengths = np.loadtxt(
        os.path.join(matches_root, 'merged_lengths.txt'),
        dtype=int)
    return mdl.Matches(indices, lengths)


if __name__ == '__main__':
    matches = load_matches(sys.argv[1])
    for dataset in progress_bar('dataset')(subdirectories(sys.argv[2])):
        data = np.load(input_filename(dataset))
        merged = mdl.apply_merging(data, matches)
        np.save(result_filename(dataset), merged)
