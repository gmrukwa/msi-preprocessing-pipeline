from functools import partial
import gc
import os

import luigi
import numpy as np
from tqdm import tqdm

from components.spectrum.alignment import pafft

from pipeline._base import *
from pipeline.baseline import RemoveBaseline
from pipeline.outlier import DetectOutliers
from pipeline.resampling import FindResamplingAxis


class ExtractPaFFTReference(ExtractReference):
    INPUT_DIR = RemoveBaseline.INPUT_DIR

    datasets = luigi.ListParameter(description="Names of the datasets to use")

    def requires(self):
        yield DetectOutliers(datasets=self.datasets, pool_size=self.pool_size)
        for dataset in self.datasets:
            yield RemoveBaseline(dataset=dataset, datasets=self.datasets,
                                 pool_size=self.pool_size)
    
    def output(self):
        return self._as_target("pafft_reference.csv")


class PaFFT(BaseTask):
    INPUT_DIR = RemoveBaseline.INPUT_DIR
    OUTPUT_DIR = os.path.join(BaseTask.OUTPUT_DIR, '04-pafft-aligned')

    dataset = luigi.Parameter(description="Dataset to align")
    datasets = luigi.ListParameter(
        description="Names of the datasets to use",
        visibility=luigi.parameter.ParameterVisibility.HIDDEN)

    def requires(self):
        yield FindResamplingAxis(datasets=self.datasets,
                                 pool_size=self.pool_size)
        yield ExtractPaFFTReference(datasets=self.datasets,
                                    pool_size=self.pool_size)
        yield RemoveBaseline(dataset=self.dataset, datasets=self.datasets,
                             pool_size=self.pool_size)

    def output(self):
        return self._as_target("{0}.npy".format(self.dataset))

    def run(self):
        mzs, reference, spectra = self.input()
        self.set_status_message('Loading data')
        mzs = np.loadtxt(mzs.path, delimiter=',')
        reference = np.loadtxt(reference.path, delimiter=',')
        spectra = np.load(spectra.path, mmap_mode='r')
        self.set_status_message('Spectra alignment')
        align = partial(pafft, mzs=mzs, reference_counts=reference)
        aligned = [
            align(spectrum) for spectrum
            in tqdm(LuigiTqdm(spectra, self), desc='Alignment')
        ]
        self.set_status_message('Saving results')
        del spectra
        gc.collect()
        with self.output().temporary_path() as tmp_path, \
                open(tmp_path, 'wb') as out_file:
            np.save(out_file, aligned)
