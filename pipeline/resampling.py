import os

import luigi
import numpy as np

from pipeline._base import AtomicTask, MakeDir, NonAtomicTask


class FindResamplingAxis(AtomicTask):
    INPUT_DIR = os.path.join(AtomicTask.INPUT_DIR, 'raw')

    datasets = luigi.ListParameter(description="Names of the datasets to use")

    def requires(self):
        return MakeDir(self.OUTPUT_DIR)

    @property
    def _output(self):
        return ['resampled_mz_axis.txt']
    
    def _run(self):
        from bin.estimate_resampling_axis import build_new_axis

        datasets = [
            os.path.join(self.INPUT_DIR, dataset)
            for dataset in self.datasets
        ]

        new_axis = build_new_axis(datasets)
        np.savetxt(os.path.join(self.OUTPUT_DIR, 'resampled_mz_axis.txt'),
                   new_axis)
