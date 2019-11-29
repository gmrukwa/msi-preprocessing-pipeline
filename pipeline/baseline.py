import os
import subprocess

import luigi

from pipeline._base import *
from pipeline.resampling import FindResamplingAxis, ResampleDataset


class RemoveBaseline(BaseTask):
    INPUT_DIR = ResampleDataset.OUTPUT_DIR
    OUTPUT_DIR = os.path.join(BaseTask.OUTPUT_DIR, '02-baseline-removed')

    dataset = luigi.Parameter(description="Dataset to remove baseline from")
    datasets = luigi.ListParameter(
        description="Names of the datasets to use",
        visibility=luigi.parameter.ParameterVisibility.HIDDEN)

    def requires(self):
        yield FindResamplingAxis(datasets=self.datasets,
                                 pool_size=self.pool_size)
        yield ResampleDataset(dataset=self.dataset, datasets=self.datasets,
                              pool_size=self.pool_size)
    
    def output(self):
        return self._as_target("{0}.npy".format(self.dataset))

    def run(self):
        mz_axis, spectra = self.input()
        with self.output().temporary_path() as tmp_path:
            subprocess.run([
                "python", "-m", "bin.baseline",
                mz_axis.path,
                spectra.path,
                tmp_path,  # destination
                str(self.pool_size)
            ], check=True, capture_output=True)


if __name__ == '__main__':
    from memory_profiler import profile
    RemoveBaseline.run = profile(RemoveBaseline.run)
    if os.path.exists('/data/02-baseline-removed/my-dataset1.npy'):
        os.remove('/data/02-baseline-removed/my-dataset1.npy')
    luigi.build([
        RemoveBaseline(dataset='my-dataset1', datasets=['my-dataset1', 'my-dataset2'])
    ], local_scheduler=True)
