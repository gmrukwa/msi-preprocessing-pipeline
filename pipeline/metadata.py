from functools import partial
import os
import re

from functional import broadcast, for_each, pipe, progress_bar
import luigi
import numpy as np
import pandas as pd

from components.io_utils import text_files

from pipeline._base import *


save_csv = partial(pd.DataFrame.to_csv, index=False)


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


class AssembleMetadata(BaseTask):
    INPUT_DIR = os.path.join(BaseTask.INPUT_DIR, 'raw')
    OUTPUT_DIR = os.path.join(BaseTask.OUTPUT_DIR, 'metadata')

    dataset = luigi.Parameter(description='Dataset to get metadata from')

    def output(self):
        return self._as_target("{0}.csv".format(self.dataset))

    def run(self):
        gather_metadata = pipe(
            text_files, list,
            partial(LuigiTqdm, task=self),
            progress_bar('gathering metadata'),
            for_each(spectrum_metadata, lazy=False),
            np.vstack,
            partial(pd.DataFrame, columns=['R', 'X', 'Y'])
        )
        data_path = os.path.join(self.INPUT_DIR, self.dataset)
        metadata = gather_metadata(data_path)
        with self.output().open('w') as outfile:
            save_csv(metadata, outfile)
