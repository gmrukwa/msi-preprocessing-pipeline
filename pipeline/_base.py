from abc import ABCMeta, abstractmethod
import logging
import os
import shutil

import luigi

logger = logging.getLogger('luigi-interface')

# The processes that will be ran have following characteristics:
# a) many directories -> few (or single) files (e.g. new m/z axis)
# b) many directories + few files -> many directories (e.g. resampling)
# 
# Writes of single files could be considered atomic, thus no magic needs
# to happen.
# 
# Writes of heavy directories are never atomic, thus we need to follow
# the guide from:
# https://luigi.readthedocs.io/en/stable/luigi_patterns.html#atomic-writes-problem
# 
# We assume, that data source will be mounted to `/data/raw` directory.
# All the outputs will be placed into the `/data` subdirectories.
# Single helper files will be placed to `/data/intermediate`.


class BaseTask(luigi.Task):
    OUTPUT_DIR = '/data'
    INPUT_DIR = '/data'

    @property
    @abstractmethod
    def _output(self):
        """Iterable of all the local paths that will contain outputs"""
        pass

    @property
    def _rooted_output(self):
        return [os.path.join(self.OUTPUT_DIR, o) for o in self._output]
    
    def output(self):
        return [luigi.LocalTarget(o) for o in self._rooted_output]


class AtomicTask(BaseTask):
    OUTPUT_DIR = os.path.join(BaseTask.OUTPUT_DIR, 'intermediate')

    # Overwrite of either ``run`` or ``_run` does the job.

    def _run(self):
        return super().run()
    
    def run(self):
        return self._run()


class NonAtomicTask(BaseTask, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._intercepted = None

    @property
    def intercepted(self):
        """Iterable of all the paths after interception"""
        if self._intercepted is None:
            self._intercepted = [
                '{0}-tmp-{1}'.format(o, self.task_id)
                for o in self._rooted_output
            ]
        return self._intercepted
    
    @abstractmethod
    def _run(self):
        """Does actual task job"""
        pass

    def run(self):
        self._run()  # run with intercepted output directories
        
        for src, dst in zip(self.intercepted, self._rooted_output):
            shutil.move(src, dst)


class MakeDir(AtomicTask):
    directory = luigi.Parameter()

    @property
    def _output(self):
        return [self.directory]
    
    def _run(self):
        os.makedirs(self.directory, exist_ok=True)


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
