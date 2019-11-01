from abc import ABCMeta, abstractmethod
import logging
import os
import shutil

import luigi

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
