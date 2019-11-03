from functools import partial

from functional import pmap
import luigi
import numpy as np
from tqdm import tqdm

from pipeline._base import *
from pipeline.alignment import PaFFT
from pipeline.outlier import DetectOutliers


class ExtractTICReference(HelperTask):
    INPUT_DIR = PaFFT.OUTPUT_DIR

    datasets = luigi.ListParameter(description="Names of the datasets to use")

    def requires(self):
        yield DetectOutliers(datasets=self.datasets)
        for dataset in self.datasets:
            yield PaFFT(dataset=dataset, datasets=self.datasets)
    
    def output(self):
        return self._as_target("tic_normalization_reference.csv")

    def run(self):
        self.set_status_message('Loading data')
        approvals, *datasets = self.input()
        approvals = [np.load(approval.path) for approval in approvals]
        self.set_status_message('Computing references')
        references = [
            np.load(spectra.path)[selection].mean(axis=0)
            for selection, spectra in zip(approvals, LuigiTqdm(datasets, self))
        ]
        counts = [np.sum(approval) for approval in approvals]
        mean = np.average(references, axis=0, weights=counts).reshape(1, -1)
        self.set_status_message('Saving results')
        with self.output().temporary_path() as tmp_path:
            np.savetxt(tmp_path, mean, delimiter=',')


def scale_to_tic(spectrum, reference_tic):
    scale = reference_tic / np.sum(spectrum)
    return spectrum * scale


class NormalizeTIC(BaseTask):
    INPUT_DIR = PaFFT.OUTPUT_DIR
    OUTPUT_DIR = os.path.join(BaseTask.OUTPUT_DIR, '05-tic-normalized')

    dataset = luigi.Parameter(description="Dataset to normalize")
    datasets = luigi.ListParameter(description="Names of the datasets to use")

    def requires(self):
        yield ExtractTICReference(datasets=self.datasets)
        yield PaFFT(dataset=self.dataset, datasets=self.datasets)
    
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
