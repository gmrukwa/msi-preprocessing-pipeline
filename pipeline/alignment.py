import os
import subprocess

import luigi

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
        with self.output().temporary_path() as tmp_path:
            subprocess.run([
                "python", "-m", "bin.alignment",
                mzs.path,
                reference.path,
                spectra.path,
                str(self.pool_size),
                tmp_path  # destination
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


if __name__ == '__main__':
    from memory_profiler import profile
    PaFFT.run = profile(PaFFT.run)
    if os.path.exists('/data/04-pafft-aligned/my-dataset1.npy'):
        os.remove('/data/04-pafft-aligned/my-dataset1.npy')
    luigi.build([
        PaFFT(dataset='my-dataset1', datasets=['my-dataset1', 'my-dataset2'])
    ], local_scheduler=True)
