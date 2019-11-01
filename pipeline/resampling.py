from functools import partial
import os

import luigi
import numpy as np

from pipeline._base import *


class FindResamplingAxis(HelperTask):
    INPUT_DIR = os.path.join(HelperTask.INPUT_DIR, 'raw')

    datasets = luigi.ListParameter(description="Names of the datasets to use")

    def output(self):
        return self._as_target('resampled_mz_axis.txt')
    
    def run(self):
        from bin.estimate_resampling_axis import build_new_axis
        datasets = [
            os.path.join(self.INPUT_DIR, dataset)
            for dataset in self.datasets
        ]
        new_axis = build_new_axis(datasets)
        with self.output().open('w') as outfile:
            np.savetxt(outfile, new_axis)


class ResampleDataset(BaseTask):
    INPUT_DIR = os.path.join(BaseTask.INPUT_DIR, 'raw')
    OUTPUT_DIR = os.path.join(BaseTask.OUTPUT_DIR, '01-resampled')

    dataset = luigi.Parameter(description="Dataset to resample")
    datasets = luigi.ListParameter(description="Names of the datasets to use")

    def requires(self):
        return FindResamplingAxis(datasets=self.datasets)
    
    def output(self):
        return self._as_target("{0}.npy".format(self.dataset))
    
    def run(self):
        from bin.resample_datasets import spectrum_sampling_pipe
        from functional import for_each, pipe, progress_bar
        from components.io_utils import text_files

        with self.input().open() as axis_file:
            new_axis = np.loadtxt(axis_file)

        dataset_sampling_pipe = pipe(
            text_files, list,
            partial(LuigiTqdm, task=self),
            progress_bar('resampling dataset'),
            for_each(spectrum_sampling_pipe(new_axis),
                     parallel=True, chunksize=800)
        )

        resampled = dataset_sampling_pipe(
            os.path.join(self.INPUT_DIR, self.dataset))

        with self.output().temporary_path() as tmp_path:
            with open(tmp_path, 'wb') as outfile:
                np.save(outfile, resampled)
