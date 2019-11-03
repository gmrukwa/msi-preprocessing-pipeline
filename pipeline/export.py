import luigi
import numpy as np

from pipeline._base import *
from pipeline.gmm import MergeDataset


class ExportCsv(BaseTask):
    INPUT_DIR = MergeDataset.OUTPUT_DIR

    dataset = luigi.Parameter(description="Dataset to export")
    datasets = luigi.ListParameter(description="Names of the datasets to use")
    fmt = luigi.Parameter(description="Format of numbers during export",
                          default='%.18e')
    
    def requires(self):
        return MergeDataset(dataset=self.dataset, datasets=self.datasets)
    
    def output(self):
        return self._as_target("{0}.csv".format(self.dataset))

    def run(self):
        self.set_status_message('Loading data')
        spectra = np.load(self.input().path)
        self.set_status_message('Exporting .csv')
        with self.output().temporary_path() as tmp_path:
            np.savetxt(tmp_path, spectra, delimiter=',', fmt=self.fmt)
