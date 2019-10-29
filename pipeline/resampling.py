import os

import luigi

from pipeline.docker import BasicDockerTask


class FindResamplingAxis(BasicDockerTask):
    intermediate_dir = luigi.Parameter()
    
    @property
    def command(self):
        return 'cp /data/LICENSE /data/out'
    
    @property
    def binds(self):
        return ['%s:/data' % self.intermediate_dir]

    def output(self):
        return luigi.LocalTarget(
            os.path.join(self.intermediate_dir, 'out'))
