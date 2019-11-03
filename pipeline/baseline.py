import os

from functional import pmap
import luigi
import numpy as np
from tqdm import tqdm

from pipeline._base import *
from pipeline.resampling import FindResamplingAxis, ResampleDataset


class RemoveBaseline(BaseTask):
    INPUT_DIR = ResampleDataset.OUTPUT_DIR
    OUTPUT_DIR = os.path.join(BaseTask.OUTPUT_DIR, '02-baseline-removed')

    dataset = luigi.Parameter(description="Dataset to remove baseline from")
    datasets = luigi.ListParameter(description="Names of the datasets to use")

    def requires(self):
        yield FindResamplingAxis(datasets=self.datasets)
        yield ResampleDataset(dataset=self.dataset, datasets=self.datasets)
    
    def output(self):
        return self._as_target("{0}.npy".format(self.dataset))

    def run(self):
        from bin.remove_baseline import baseline_remover
        self.set_status_message('Loading data')
        mz_axis, spectra = self.input()
        mz_axis = np.loadtxt(mz_axis.path, delimiter=',')
        remover = baseline_remover(mz_axis)
        spectra = np.load(spectra.path)
        self.set_status_message('Removing baseline')
        lowered = pmap(remover,
                       tqdm(LuigiTqdm(spectra, self),
                            desc='Baseline removal'),
                       chunksize=800)
        self.set_status_message('Saving result')
        with self.output().temporary_path() as tmp_path:
            with open(tmp_path, 'wb') as outfile:
                np.save(outfile, lowered)


if __name__ == '__main__':
    luigi.build([RemoveBaseline()], local_scheduler=True)
