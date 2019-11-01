from functools import partial

import numpy as np
from tqdm import tqdm

from components.spectrum.alignment import pafft

from pipeline._base import *
from pipeline.baseline import RemoveBaseline
from pipeline.outlier import DetectOutliers
from pipeline.resampling import FindResamplingAxis


class ExtractReference(HelperTask):
    INPUT_DIR = RemoveBaseline.INPUT_DIR

    datasets = luigi.ListParameter(description="Names of the datasets to use")

    def requires(self):
        yield DetectOutliers(datasets=self.datasets)
        for dataset in self.datasets:
            yield RemoveBaseline(dataset=dataset, datasets=self.datasets)
    
    def output(self):
        return self._as_target("pafft_reference.csv")

    def run(self):
        approvals, *datasets = self.input()
        approvals = [np.load(approval.path) for approval in approvals]
        references = [
            np.load(spectra.path)[selection].mean(axis=0)
            for selection, spectra in zip(approvals, LuigiTqdm(datasets, self))
        ]
        counts = [np.sum(approval) for approval in approvals]
        mean = np.average(references, axis=0, weights=counts).reshape(1, -1)
        with self.output().temporary_path() as tmp_path:
            np.savetxt(tmp_path, mean, delimiter=',')


class PaFFT(BaseTask):
    INPUT_DIR = RemoveBaseline.INPUT_DIR
    OUTPUT_DIR = os.path.join(BaseTask.OUTPUT_DIR, '04-pafft-aligned')

    dataset = luigi.Parameter(description="Dataset to align")
    datasets = luigi.ListParameter(description="Names of the datasets to use")

    def requires(self):
        yield FindResamplingAxis(datasets=self.datasets)
        yield ExtractReference(datasets=self.datasets)
        yield RemoveBaseline(dataset=self.dataset, datasets=self.datasets)

    def output(self):
        return self._as_target("{0}.npy".format(self.dataset))

    def run(self):
        mzs, reference, spectra = self.input()
        mzs = np.loadtxt(mzs.path, delimiter=',')
        reference = np.loadtxt(reference.path, delimiter=',')
        spectra = np.load(spectra.path)
        align = partial(pafft, mzs=mzs, reference_counts=reference)
        aligned = [
            align(spectrum) for spectrum
            in tqdm(LuigiTqdm(spectra, self), desc='Alignment')
        ]
        with self.output().temporary_path() as tmp_path, \
                open(tmp_path, 'wb') as out_file:
            np.save(out_file, aligned)
