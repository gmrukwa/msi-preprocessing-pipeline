# -*- coding: utf-8 -*-
"""
Created on Sat May 19 17:30:28 2018

Find new common m/z axis. Path to datasets root is the parameter to the script.

1) Estimate widest common m/z range
2) Estimate smallest number of samples in the common range
3) Estimate new m/z axis on first preparation (differences are negligible)

@author: Grzegorz Mrukwa
"""

from functools import partial, reduce
import os
import sys

import numpy as np

from functional import as_arguments_of, broadcast, for_each, pipe, report_value, take
from components.io_utils import subdirectories, text_files, try_loadtxt
from components.spectrum.resampling import estimate_new_axis


# ==============================================================================
#                               COMMON RANGE
# ==============================================================================


def get_mzs_from_content(content: np.ndarray) -> np.ndarray:
    return content[:, 0]


get_mz_axis = pipe(
    text_files,
    partial(take, n_elements=1),
    as_arguments_of(try_loadtxt),
    get_mzs_from_content
)


mz_range = pipe(
    report_value('dataset'),
    get_mz_axis,
    broadcast(np.min, np.max),
    np.array,
    report_value('mz range')
)


def common_part(first, second):
    return max(first[0], second[0]), min(first[1], second[1])


datasets = pipe(subdirectories, list)


common_range = pipe(
    for_each(mz_range),
    partial(reduce, common_part)
)


# ==============================================================================
#                          SMALLEST NUMBER OF SAMPLES
# ==============================================================================


def count_samples_in_common_range(mzs, common_mzs):
    return np.sum(np.logical_and(common_mzs[0] <= mzs, mzs <= common_mzs[1]))


def smallest_number_of_samples(found_datasets):
    common_mzs = common_range(found_datasets)
    count_mzs = partial(count_samples_in_common_range, common_mzs=common_mzs)
    estimate = pipe(
        for_each(pipe(get_mz_axis, count_mzs), lazy=False),
        report_value('mzs amounts'),
        np.min,
        report_value('minimal number of mzs')
    )
    return estimate(found_datasets)


# ==============================================================================
#                              RESAMPLING AXIS
# ==============================================================================


get_some_axis = pipe(
    partial(take, n_elements=1),
    as_arguments_of(get_mz_axis)
)


build_new_axis = pipe(
    broadcast(
        get_some_axis,
        smallest_number_of_samples,
        pipe(common_range, report_value('common range'))
    ),
    as_arguments_of(estimate_new_axis)
)


# ==============================================================================
#                                   MAIN
# ==============================================================================


def main():
    new_axis = build_new_axis(datasets(sys.argv[1]))
    np.savetxt(os.path.join(sys.argv[2], 'new_mz_axis.txt'), new_axis)


if __name__ == '__main__':
    main()
