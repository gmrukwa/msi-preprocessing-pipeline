import gc
import os

import luigi
import numpy as np

from components.spectrum.outlier import detect
from pipeline._base import *
from pipeline.baseline import RemoveBaseline


def _scatter(col, like):
    limits = np.cumsum([0] + [v.size for v in like], dtype=int)
    scattered = [col[start:end] for start, end in zip(limits[:-1], limits[1:])]
    assert all(chunk1.size == chunk2.size for chunk1, chunk2 in
                zip(like, scattered))
    return scattered


class DetectOutliers(BaseTask):
    INPUT_DIR = RemoveBaseline.OUTPUT_DIR
    OUTPUT_DIR = os.path.join(BaseTask.OUTPUT_DIR, '03-outlier-filter')

    datasets = luigi.ListParameter(description="Names of the datasets to use")

    def requires(self):
        return [
            RemoveBaseline(dataset=dataset, datasets=self.datasets,
                           pool_size=self.pool_size)
            for dataset in self.datasets
        ]
    
    def output(self):
        return [
            self._as_target("{0}.npy".format(dataset))
            for dataset in self.datasets
        ]
    
    def run(self):
        self.set_status_message('Loading data')
        tics = [
            np.load(spectra.path, mmap_mode='r').sum(axis=1)
            for spectra in LuigiTqdm(self.input(), self)
        ]
        self.set_status_message('Outlier detection')
        outliers = detect(np.hstack(tics))
        self.set_status_message('Saving results')
        chunk_outliers = _scatter(outliers, like=tics)
        for dataset, detection in zip(self.datasets, chunk_outliers):
            logger.info(
                'Preserving {0} spectra out of {1} for {2}, removed '
                '{3} outliers'.format(
                    np.sum(np.logical_not(detection)), detection.size,
                    dataset, np.sum(detection)))
        for detection, output in zip(chunk_outliers, self.output()):
            with output.temporary_path() as tmp_path, \
                    open(tmp_path, 'wb') as out_file:
                np.save(out_file, np.logical_not(detection))
