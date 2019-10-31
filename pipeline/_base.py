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
        if all(o.exists() for o in self.output()):  # idempotency
            return
        
        for path in self.intercepted:
            os.makedirs(path)
        
        self._run()  # run with intercepted output directories
        
        for src, dst in zip(self.intercepted, self._output):
            shutil.move(src, dst)


class MakeIntermediateDir(AtomicTask):
    @property
    def _output(self):
        return [self.OUTPUT_DIR]
    
    def _run(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
