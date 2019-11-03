import logging
import os

import luigi
import numpy as np


__all__ = [
    'logger',
    'BaseTask',
    'HelperTask',
    'LuigiTqdm',
    'ExtractReference'
]


logger = logging.getLogger('luigi-interface')

# We assume, that data source will be mounted to `/data/raw` directory.
# All the outputs will be placed into the `/data` subdirectories.
# Single helper files will be placed to `/data/intermediate`.


class BaseTask(luigi.Task):
    OUTPUT_DIR = '/data'
    INPUT_DIR = '/data'

    def _as_target(self, fname: str):
        return luigi.LocalTarget(os.path.join(self.OUTPUT_DIR, fname))


class HelperTask(BaseTask):
    OUTPUT_DIR = os.path.join(BaseTask.OUTPUT_DIR, 'intermediate')


class LuigiTqdm:
    def __init__(self, col, task: luigi.Task):
        self.col = col
        self.task = task
    
    def __iter__(self):
        self.task.set_status_message('in progress')
        min_increment = len(self.col) // 100 or 1
        for idx, item in enumerate(self.col):
            yield item
            if idx % min_increment == 0:
                percentage = (100 * idx) // len(self.col)
                self.task.set_progress_percentage(percentage)
        self.task.set_progress_percentage(100)
        self.task.set_status_message('finished')

    def __len__(self):
        return len(self.col)


class ExtractReference(HelperTask):
    """Base class for extracting mean spectrum

    Descends must implement ``requires``, ``output``.
    ``requires`` is supposed to have outlier detection + data stages
    ``output`` is supposed to point a single file destination path
    """
    def run(self):
        self.set_status_message('Loading data')
        approvals, *datasets = self.input()
        approvals = [np.load(approval.path) for approval in approvals]
        self.set_status_message('Computing references')
        references = [
            np.load(spectra.path)[selection].mean(axis=0)
            for selection, spectra in zip(approvals, LuigiTqdm(datasets, self))
        ]
        counts = [np.sum(approval) for approval in approvals]
        mean = np.average(references, axis=0, weights=counts).reshape(1, -1)
        self.set_status_message('Saving results')
        with self.output().temporary_path() as tmp_path:
            np.savetxt(tmp_path, mean, delimiter=',')
