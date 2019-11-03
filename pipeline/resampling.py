from functools import partial
import os

from functional import as_arguments_of, for_each, pipe, progress_bar, tee
import luigi
import numpy as np

from components.io_utils import text_files, try_loadtxt

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
        new_axis = build_new_axis(datasets).reshape(1, -1)
        with self.output().open('w') as outfile:
            np.savetxt(outfile, new_axis, delimiter=',')


def spectrum_sampling_pipe(mzs):
    return pipe(
        try_loadtxt,
        np.transpose,
        as_arguments_of(partial(np.interp, mzs)),
        np.ravel,
        partial(np.ndarray.astype, dtype=np.float32)
    )


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
        self.set_status_message('Loading data')
        new_axis = np.loadtxt(self.input().path, delimiter=',')

        dataset_sampling_pipe = pipe(
            text_files, list,
            tee(lambda _: self.set_status_message('Resampling dataset')),
            partial(LuigiTqdm, task=self),
            progress_bar('resampling dataset'),
            for_each(spectrum_sampling_pipe(new_axis),
                     parallel=True, chunksize=800)
        )

        resampled = dataset_sampling_pipe(
            os.path.join(self.INPUT_DIR, self.dataset))

        self.set_status_message('Saving results')
        with self.output().temporary_path() as tmp_path:
            with open(tmp_path, 'wb') as outfile:
                np.save(outfile, resampled)
