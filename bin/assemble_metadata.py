# -*- coding: utf-8 -*-
"""Gather metadata of spectra in datasets.

Created on Sat May 19 17:30:28 2018

Datasets should be passed as consecutive parameters to the script.
Metadata of each dataset will be saved to the same directory under
'metadata.csv'.

@author: Grzegorz Mrukwa
"""

from functools import partial
import os
import re
import sys

import numpy as np
import pandas as pd

from functional import as_arguments_of, broadcast, for_each, pipe, progress_bar

from components.io_utils import text_files


metadata_pattern = re.compile('_R[0-9]+X[0-9]+Y[0-9]+_')


def component_length(component_with_suffix):
    for idx, element in enumerate(component_with_suffix):
        if not element.isdigit():
            return idx
    return len(component_with_suffix)


def get_component(filename, component):
    match = metadata_pattern.search(filename)
    if match is None:
        raise ValueError(filename)
    metadata = filename[match.start() + 1:match.end() - 1]
    metadata = metadata[metadata.find(component) + 1:]
    metadata = metadata[:component_length(metadata)]
    return int(metadata)


spectrum_metadata = pipe(
    broadcast(
        partial(get_component, component='R'),
        partial(get_component, component='X'),
        partial(get_component, component='Y')
    ),
    np.array
)


def metadata_filename(dataset_path):
    return os.path.join(dataset_path, 'metadata.csv')


gather_metadata = pipe(
    text_files, list,
    progress_bar('gathering metadata'),
    for_each(spectrum_metadata),
    np.vstack,
    partial(pd.DataFrame, columns=['R', 'X', 'Y'])
)


save_csv = partial(pd.DataFrame.to_csv, index=False)


gather_and_save_metadata = pipe(
    broadcast(
        gather_metadata,
        metadata_filename
    ),
    as_arguments_of(save_csv)
)


assemble_metadata = pipe(
    progress_bar('processing datasets'),
    for_each(gather_and_save_metadata, lazy=False)
)


if __name__ == '__main__':
    assemble_metadata(sys.argv[1:])
