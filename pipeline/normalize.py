from functools import partial
import os

from functional import pmap
import luigi
import numpy as np
from tqdm import tqdm

from pipeline._base import *
from pipeline.alignment import PaFFT
from pipeline.outlier import DetectOutliers


class ExtractTICReference(ExtractReference):
    INPUT_DIR = PaFFT.OUTPUT_DIR

    datasets = luigi.ListParameter(description="Names of the datasets to use")

    def requires(self):
        yield DetectOutliers(datasets=self.datasets, pool_size=self.pool_size)
        for dataset in self.datasets:
            yield PaFFT(dataset=dataset, datasets=self.datasets,
                        pool_size=self.pool_size)
    
    def output(self):
        return self._as_target("tic_normalization_reference.csv")


def scale_to_tic(spectrum, reference_tic):
    scale = reference_tic / np.sum(spectrum)
    return spectrum * scale


class NormalizeTIC(BaseTask):
    INPUT_DIR = PaFFT.OUTPUT_DIR
    OUTPUT_DIR = os.path.join(BaseTask.OUTPUT_DIR, '05-tic-normalized')

    dataset = luigi.Parameter(description="Dataset to normalize")
    datasets = luigi.ListParameter(
        description="Names of the datasets to use",
        visibility=luigi.parameter.ParameterVisibility.HIDDEN)

    def requires(self):
        yield ExtractTICReference(datasets=self.datasets,
                                  pool_size=self.pool_size)
        yield PaFFT(dataset=self.dataset, datasets=self.datasets,
                    pool_size=self.pool_size)
    
    def output(self):
        return self._as_target("{0}.npy".format(self.dataset))
    
    def run(self):
        self.set_status_message('Loading data')
        reference, spectra = self.input()
        reference = np.loadtxt(reference.path, delimiter=',')
        spectra = np.load(spectra.path)
        self.set_status_message('Computing reference TIC')
        reference_tic = np.sum(reference)
        logger.info("Reference TIC: {0}".format(reference_tic))
        self.set_status_message('Normalizing spectra')
        rescale = partial(scale_to_tic, reference_tic=reference_tic)
        spectra = LuigiTqdm(spectra, self)
        spectra = tqdm(spectra, desc='TIC normalization')
        normalized = pmap(rescale, spectra, chunksize=800)
        self.set_status_message('Saving results')
        with self.output().temporary_path() as tmp_path, \
                open(tmp_path, 'wb') as out_file:
            np.save(out_file, normalized)
