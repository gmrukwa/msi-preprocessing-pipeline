from functional import pmap
import luigi
import numpy as np
from tqdm import tqdm

from pipeline._base import *
from pipeline.resampling import FindResamplingAxis, ResampleDataset


class RemoveBaseline(NonAtomicTask):
    INPUT_DIR = ResampleDataset.OUTPUT_DIR
    OUTPUT_DIR = os.path.join(NonAtomicTask.OUTPUT_DIR, 'baseline-removed')

    dataset = luigi.Parameter(description="Dataset to remove baseline from")
    datasets = luigi.ListParameter(description="Names of the datasets to use")

    def requires(self):
        yield MakeDir(self.OUTPUT_DIR)
        yield FindResamplingAxis(datasets=self.datasets)
        yield ResampleDataset(dataset=self.dataset, datasets=self.datasets)
    
    @property
    def _output(self):
        return ["{0}.npy".format(self.dataset)]

    def _run(self):
        from bin.remove_baseline import baseline_remover
        with self.input()[1][0].open('r') as infile:
            mz_axis = np.loadtxt(infile)
        remover = baseline_remover(mz_axis)
        with self.input()[2][0].open('rb') as infile:
            spectra = np.load(infile)
        lowered = pmap(remover,
                       tqdm(LuigiTqdm(spectra, self),
                            desc='Baseline removal'),
                       chunksize=800)
        with open(self.intercepted[0], 'wb') as outfile:
            np.save(outfile, lowered)
