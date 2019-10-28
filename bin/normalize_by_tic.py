"""
Arguments:
    path to reference spectrum
    path to datasets root

"""
from functools import partial
import os
import sys

from functional import for_each, pipe, progress_bar
import numpy as np
from tqdm import tqdm

from components.utils import subdirectories


def scale_to_tic(spectrum, reference_tic):
    scale = reference_tic / np.sum(spectrum)
    return spectrum * scale


def input_filename(dataset_root):
    return os.path.join(dataset_root, 'aligned.npy')


def result_filename(dataset_root):
    return os.path.join(dataset_root, 'normalized.npy')


if __name__ == '__main__':
    reference_spectrum = np.loadtxt(sys.argv[1])
    reference_tic = np.sum(reference_spectrum)
    tqdm.write('Reference TIC: {0}'.format(reference_tic))

    rescale_spectrum = partial(scale_to_tic, reference_tic=reference_tic)

    rescale_dataset = pipe(
        input_filename,
        np.load,
        progress_bar('Spectrum'),
        for_each(rescale_spectrum, parallel=True)
    )

    for dataset_root in tqdm(subdirectories(sys.argv[2])):
        result = rescale_dataset(dataset_root)
        np.save(result_filename(dataset_root), result)
