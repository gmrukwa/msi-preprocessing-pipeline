import os

import luigi

from pipeline._base import AtomicTask, MakeIntermediateDir, NonAtomicTask


class FindResamplingAxis(AtomicTask):
    INPUT_DIR = os.path.join(AtomicTask.INPUT_DIR, 'raw')

    def requires(self):
        return MakeIntermediateDir()

    @property
    def _output(self):
        return ['resampled_mz_axis.txt']
    
    def _run(self):
        with open(self._rooted_output[0], 'w') as outfile:
            outfile.write("Hello World!")
