from functools import partial
import os
import sys

from functional import for_each, pipe, progress_bar
import numpy as np
from tqdm import tqdm

from components.spectrum.outlier import detect
from components.utils import subdirectories


def result_path(dataset_root):
    return os.path.join(dataset_root, 'outliers.npy')


def input_filename(dataset_root):
    return os.path.join(dataset_root, 'lowered.npy')


gather_tics = pipe(input_filename, np.load, partial(np.sum, axis=1))
gather_all_tics = pipe(subdirectories,
                       progress_bar('Gathering TIC'),
                       for_each(gather_tics, lazy=False))

if __name__ == '__main__':
    if not os.path.exists('outliers.npy'):
        tics = gather_all_tics(sys.argv[1])
        outliers = detect(np.hstack(tics))
        np.save('outliers.npy', outliers)
        np.save('tics.npy', tics)
    else:
        outliers = np.load('outliers.npy')
        tics = np.load('tics.npy')

    limits = np.cumsum([0] + [part.size for part in tics], dtype=int)
    chunk_outliers = [
        outliers[start:end] for start, end in zip(limits[:-1], limits[1:])
    ]
    assert all(chunk1.size == chunk2.size for chunk1, chunk2 in
               zip(tics, chunk_outliers))
    dataset_roots = tqdm(subdirectories(sys.argv[1]), desc='Filtering')
    for dataset_root, binary_filter in zip(dataset_roots, chunk_outliers):
        dataset = input_filename(dataset_root)
        allowed_spectra = np.logical_not(binary_filter)
        report = 'Preserving {0} spectra out of {1} for {2}'.format(
            np.sum(allowed_spectra),
            allowed_spectra.size,
            dataset
        )
        tqdm.write(report)
        np.save(result_path(dataset_root), binary_filter)
