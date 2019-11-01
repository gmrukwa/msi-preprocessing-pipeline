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


class ResampleDataset(NonAtomicTask):
    INPUT_DIR = os.path.join(NonAtomicTask.INPUT_DIR, 'raw')
    OUTPUT_DIR = os.path.join(NonAtomicTask.OUTPUT_DIR, 'resampled')

    dataset = luigi.Parameter(description="Dataset to resample")

    def requires(self):
        return MakeDir(self.OUTPUT_DIR), FindResamplingAxis()
    
    @property
    def _output(self):
        return ["{0}.npy".format(self.dataset)]
    
    def _run(self):
        from bin.resample_datasets import dataset_sampling_pipe
        axis_path = os.path.join(AtomicTask.OUTPUT_DIR,
                                 'resampled_mz_axis.txt')
        new_axis = np.loadtxt(axis_path)
        resampled = dataset_sampling_pipe(new_axis)(
            os.path.join(self.INPUT_DIR, self.dataset))
        with open(self.intercepted[0], 'wb') as outfile:
            np.save(outfile, resampled)  # otherwise `.npy` is suffixed
