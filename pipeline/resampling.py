from functools import partial, reduce
import os

from functional import (
    as_arguments_of,
    broadcast,
    for_each,
    pipe,
    progress_bar,
    report_value,
    take,
    tee,
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
    datasets = luigi.ListParameter(
        description="Names of the datasets to use",
        visibility=luigi.parameter.ParameterVisibility.HIDDEN)

    def requires(self):
        return FindResamplingAxis(datasets=self.datasets,
                                  pool_size=self.pool_size)
    
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
