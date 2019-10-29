"""Merge datasets into one

Arguments:
    dataset suffix
    outliers suffix
    metadata suffix
    mz axis path
    config path
    datasets root
    destination directory

"""
from functools import partial
import json
import os
import shutil
import sys
from typing import List, NamedTuple, Tuple

import pandas as pd
import numpy as np

import components.utils as utils


Arguments = NamedTuple('Arguments', [
    ('dataset_suffix', str),
    ('outliers_suffix', str),
    ('metadata_suffix', str),
    ('mz_axis_path', str),
    ('config_path', str),
    ('datasets_root', str),
    ('destination_directory', str)
])


Dataset = NamedTuple('Dataset', [
    ('spectra', np.ndarray),
    ('metadata', pd.DataFrame),
])


def load_dataset(dataset_root: str, arguments: Arguments) -> Dataset:
    spectra = np.load(os.path.join(dataset_root, arguments.dataset_suffix))
    metadata = pd.read_csv(
        os.path.join(dataset_root, arguments.metadata_suffix),
        index_col=False)
    outliers = np.load(os.path.join(dataset_root, arguments.outliers_suffix))
    return Dataset(spectra=spectra[~outliers], metadata=metadata[~outliers])


IdentifiedDataset = Tuple[Dataset, int]


def merge_datasets(datasets: List[IdentifiedDataset], shift: int=10) -> Dataset:
    # add identification to metadata
    for dataset, identification in datasets:
        dataset.metadata['dataset'] = identification
    # reorganize coordinates
    max_known_y = 0
    for dataset, _ in datasets:
        dataset.metadata['Y'] += max_known_y + shift
        max_known_y = dataset.metadata['Y'].max()
    spectra, metadata = zip(*[dataset for dataset, _ in datasets])
    all_spectra = np.vstack(spectra)
    all_metadata = pd.concat(metadata, ignore_index=True, copy=False)
    return Dataset(spectra=all_spectra, metadata=all_metadata)


def save_dataset(dataset: Dataset, arguments: Arguments):
    os.makedirs(arguments.destination_directory)
    fname = partial(os.path.join, arguments.destination_directory)
    np.save(fname('spectra.npy'), dataset.spectra)
    dataset.metadata.to_csv(fname('metadata.csv'), index=False)
    shutil.copy(arguments.mz_axis_path, fname('mz.txt'))


if __name__ == '__main__':
    arguments = Arguments(*sys.argv[1:])
    load = partial(load_dataset, arguments=arguments)
    with open(arguments.config_path) as config:
        identification = json.load(config)
    datasets = [
        (load(path), identification[path])
        for path in utils.subdirectories(arguments.datasets_root)
        if path in identification
    ]
    whole_dataset = merge_datasets(datasets)
    save_dataset(whole_dataset, arguments)
