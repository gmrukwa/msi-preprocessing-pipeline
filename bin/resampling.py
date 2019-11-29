from functools import partial
import gc
import sys

from functional import (
    as_arguments_of,
    for_each,
    pipe,
    progress_bar,
)
import numpy as np

from components.io_utils import text_files, try_loadtxt


def spectrum_sampling_pipe(mzs):
    return pipe(
        try_loadtxt,
        np.transpose,
        as_arguments_of(partial(np.interp, mzs)),
        np.ravel,
        partial(np.ndarray.astype, dtype=np.float32)
    )


if __name__ == '__main__':
    DATASET = sys.argv[1]
    MZ_AXIS = sys.argv[2]
    DESTINATION = sys.argv[3]
    new_axis = np.loadtxt(MZ_AXIS, delimiter=',')
    resampled = pipe(
        text_files, list,
        progress_bar('resampling dataset'),
        for_each(spectrum_sampling_pipe(new_axis),
                 parallel=True, chunksize=800)
    )(DATASET)
    with open(DESTINATION, 'wb') as outfile:
        np.save(outfile, resampled)
