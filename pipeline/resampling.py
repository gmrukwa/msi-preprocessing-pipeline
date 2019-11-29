from functools import partial, reduce
import gc
import os
import subprocess

from functional import (
    as_arguments_of,
    broadcast,
    for_each,
    pipe,
    report_value,
    take,
)
import luigi
import numpy as np

from components.io_utils import text_files, try_loadtxt
from components.spectrum.resampling import estimate_new_axis
from pipeline._base import *


def get_mzs_from_content(content: np.ndarray) -> np.ndarray:
    return content[:, 0]


get_mz_axis = pipe(
    text_files,
    partial(take, n_elements=1),
    as_arguments_of(try_loadtxt),
    get_mzs_from_content
)


mz_range = pipe(
    report_value('dataset'),
    get_mz_axis,
    broadcast(np.min, np.max),
    np.array,
    report_value('mz range')
)


def common_part(first, second):
    return max(first[0], second[0]), min(first[1], second[1])


common_range = pipe(
    for_each(mz_range),
    partial(reduce, common_part)
)


def count_samples_in_common_range(mzs, common_mzs):
    return np.sum(np.logical_and(common_mzs[0] <= mzs, mzs <= common_mzs[1]))


def smallest_number_of_samples(found_datasets):
    common_mzs = common_range(found_datasets)
    count_mzs = partial(count_samples_in_common_range, common_mzs=common_mzs)
    estimate = pipe(
        for_each(pipe(get_mz_axis, count_mzs), lazy=False),
        report_value('mzs amounts'),
        np.min,
        report_value('minimal number of mzs')
    )
    return estimate(found_datasets)


get_some_axis = pipe(
    partial(take, n_elements=1),
    as_arguments_of(get_mz_axis)
)


build_new_axis = pipe(
    broadcast(
        get_some_axis,
        smallest_number_of_samples,
        pipe(common_range, report_value('common range'))
    ),
    as_arguments_of(estimate_new_axis)
)


class FindResamplingAxis(HelperTask):
    INPUT_DIR = os.path.join(HelperTask.INPUT_DIR, 'raw')

    datasets = luigi.ListParameter(description="Names of the datasets to use")

    def output(self):
        return self._as_target('resampled_mz_axis.txt')
    
    def run(self):
        datasets = [
            os.path.join(self.INPUT_DIR, dataset)
            for dataset in self.datasets
        ]
        new_axis = build_new_axis(datasets).reshape(1, -1)
        with self.output().open('w') as outfile:
            np.savetxt(outfile, new_axis, delimiter=',')


class ResampleDataset(BaseTask):
    INPUT_DIR = os.path.join(BaseTask.INPUT_DIR, 'raw')
    OUTPUT_DIR = os.path.join(BaseTask.OUTPUT_DIR, '01-resampled')

    dataset = luigi.Parameter(description="Dataset to resample")
    datasets = luigi.ListParameter(
        description="Names of the datasets to use",
        visibility=luigi.parameter.ParameterVisibility.HIDDEN)

    def requires(self):
        return FindResamplingAxis(datasets=self.datasets,
                                  pool_size=self.pool_size)
    
    def output(self):
        return self._as_target("{0}.npy".format(self.dataset))
    
    def run(self):
        with self.output().temporary_path() as tmp_path:
            subprocess.run([
                "/usr/local/bin/python", "-m", "bin.resampling",
                os.path.join(self.INPUT_DIR, self.dataset),  # dataset path
                self.input().path,  # mz axis path
                tmp_path  # destination path
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


if __name__ == '__main__':
    from memory_profiler import profile
    ResampleDataset.run = profile(ResampleDataset.run)
    if os.path.exists('/data/01-resampled/my-dataset1.npy'):
        os.remove('/data/01-resampled/my-dataset1.npy')
    luigi.build([
        ResampleDataset(dataset='my-dataset1', datasets=['my-dataset1', 'my-dataset2'])
    ], local_scheduler=True)
