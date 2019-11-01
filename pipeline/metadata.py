from functools import partial
import os

import luigi
import numpy as np
import pandas as pd

from pipeline._base import *


class AssembleMetadata(BaseTask):
    INPUT_DIR = os.path.join(BaseTask.INPUT_DIR, 'raw')
    OUTPUT_DIR = os.path.join(BaseTask.OUTPUT_DIR, 'metadata')

    dataset = luigi.Parameter(description='Dataset to get metadata from')

    def output(self):
        return self._as_target("{0}.csv".format(self.dataset))

    def run(self):
        from components.io_utils import text_files
        from bin.assemble_metadata import (
            spectrum_metadata,
            save_csv,
        )
        from functional import (
            for_each,
            pipe,
            progress_bar,
        )
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
