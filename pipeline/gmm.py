from contextlib import contextmanager
from functools import partial
import json
import platform

from functional import pmap
import luigi
import numpy as np
from tqdm import tqdm

from components.spectrum.resampling import estimate_new_axis
from pipeline._base import *
from pipeline.normalize import NormalizeTIC
from pipeline.outlier import DetectOutliers
from pipeline.resampling import FindResamplingAxis


class ExtractGMMReference(HelperTask):
    INPUT_DIR = NormalizeTIC.OUTPUT_DIR

    datasets = luigi.ListParameter(description="Names of the datasets to use")

    def requires(self):
        yield DetectOutliers(datasets=self.datasets)
        for dataset in self.datasets:
            yield NormalizeTIC(dataset=dataset, datasets=self.datasets)
    
    def output(self):
        return self._as_target("gmm_reference.csv")

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


resample = np.interp


EMPIRICAL_OPTIMAL_CHANNELS_NUMBER = 100000


class ResampleGMMReference(HelperTask):
    INPUT_DIR = HelperTask.OUTPUT_DIR

    datasets = luigi.ListParameter(description="Names of the datasets to use")

    def requires(self):
        yield FindResamplingAxis(datasets=self.datasets)
        yield ExtractGMMReference(datasets=self.datasets)

    def output(self):
        yield self._as_target('resampled_gmm_reference.csv')
        yield self._as_target('resampled_gmm_reference_mz_axis.csv')

    def run(self):
        old_mzs, old_reference = self.input()
        reference_dst, new_mzs_dst = self.output()

        old_mzs = np.loadtxt(old_mzs.path, delimiter=',')
        old_reference = np.loadtxt(old_reference.path, delimiter=',')

        limits = np.min(old_mzs), np.max(old_mzs)
        new_mzs = estimate_new_axis(
            old_axis=old_mzs,
            number_of_ticks=EMPIRICAL_OPTIMAL_CHANNELS_NUMBER,
            axis_limits=limits
        )

        with new_mzs_dst.temporary_path() as tmp_path:
            np.savetxt(tmp_path, new_mzs.reshape(1, -1), delimiter=',')
        
        resampled = resample(new_mzs, old_mzs, old_reference)

        with reference_dst.temporary_path() as tmp_path:
            np.savetxt(tmp_path, resampled.reshape(1, -1), delimiter=',')


_MATLAB_SEARCH_PATHS = \
    "/usr/local/MATLAB/MATLAB_Runtime/v91/runtime/glnxa64:" + \
    "/usr/local/MATLAB/MATLAB_Runtime/v91/bin/glnxa64:" + \
    "/usr/local/MATLAB/MATLAB_Runtime/v91/sys/os/glnxa64:" + \
    "/usr/local/MATLAB/MATLAB_Runtime/v91/sys/opengl/lib/glnxa64:"


_local_system = platform.system()

if _local_system == 'Windows':
    # Must be here. Doesn't work as contextmanager.
    # If you think different increase counter of wasted hours: 4
    os.environ['PATH'] = os.environ['PATH'].lower()


@contextmanager
def _matlab_paths():
    if _local_system == 'Linux':
        old_env = os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['LD_LIBRARY_PATH'] = _MATLAB_SEARCH_PATHS + old_env
    elif _local_system == 'Darwin':
        raise NotImplementedError('OSX hosts are not supported.')
    try:
        yield
    finally:
        if _local_system == 'Linux':
            os.environ['LD_LIBRARY_PATH'] = old_env


class BuildGMM(HelperTask):
    INPUT_DIR = HelperTask.INPUT_DIR

    datasets = luigi.ListParameter(description="Names of the datasets to use")

    def requires(self):
        return ResampleGMMReference(datasets=self.datasets)
    
    def output(self):
        yield self._as_target('mu.csv')
        yield self._as_target('sig.csv')
        yield self._as_target('w.csv')
        yield self._as_target('gmm_model.json')
    
    def run(self):
        spectrum, mzs = self.input()
        mu_dst, sig_dst, w_dst, gmm_dst = self.output()
        
        spectrum = np.loadtxt(spectrum.path, delimiter=',')
        mzs = np.loadtxt(mzs.path, delimiter=',')
        
        with _matlab_paths():
            import MatlabAlgorithms.MsiAlgorithms as msi
            import matlab

            def as_matlab_type(array):
                return matlab.double([list(map(float, array.ravel()))])

            self.set_status_message('MCR initialization')
            logger.info('MCR initialization')
            engine = msi.initialize()
            
            spectrum = as_matlab_type(spectrum)
            mzs = as_matlab_type(mzs)
            
            self.set_status_message('Model construction')
            logger.info('Model construction')
            model = engine.estimate_gmm(mzs, spectrum, nargout=1)
            mu = np.array(model['mu'], dtype=float).ravel()
            sig = np.array(model['sig'], dtype=float).ravel()
            w = np.array(model['w'], dtype=float).ravel()
            model = {
                    'mu': list(mu),
                    'sig': list(sig),
                    'w': list(w),
                    'KS': int(model['KS']),
                    'meanspec': list(np.array(model['meanspec'],
                                    dtype=float).ravel())
            }

            # The line below fails due to unknown reasons, but MathWorks
            # won't fix it anyway.
            # engine.close()

        with mu_dst.temporary_path() as tmp_path:
            np.savetxt(tmp_path, mu, delimiter=',')
        with sig_dst.temporary_path() as tmp_path:
            np.savetxt(tmp_path, sig, delimiter=',')
        with w_dst.temporary_path() as tmp_path:
            np.savetxt(tmp_path, w, delimiter=',')
        with gmm_dst.temporary_path() as tmp_path, \
                open(tmp_path, 'w') as out_file:
            json.dump(model, out_file, indent=2, sort_keys=True)
