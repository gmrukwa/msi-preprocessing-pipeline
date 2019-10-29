import os

import luigi

from pipeline.docker import BasicDockerTask


class FindResamplingAxis(BasicDockerTask):
    @property
    def command(self):
        return 'python -m bin.estimate_resampling_axis /data /intermediate'

    def output(self):
        return luigi.LocalTarget(
            os.path.join(self.intermediate_dir, 'new_mz_axis.txt'))
