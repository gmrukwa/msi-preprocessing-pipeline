from contextlib import contextmanager
from functools import partial
import json
import platform

from functional import pmap
import luigi
import numpy as np
from scipy.stats import norm
from tqdm import tqdm

from components.spectrum.resampling import estimate_new_axis
from components.matlab_legacy import estimate_gmm, find_thresholds
from components.stats import matlab_alike_quantile
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
        
        mu, sig, w, model = estimate_gmm(mzs, spectrum)

        logger.info('Found {0} GMM components'.format(mu.size))

        with mu_dst.temporary_path() as tmp_path:
            np.savetxt(tmp_path, mu, delimiter=',')
        with sig_dst.temporary_path() as tmp_path:
            np.savetxt(tmp_path, sig, delimiter=',')
        with w_dst.temporary_path() as tmp_path:
            np.savetxt(tmp_path, w, delimiter=',')
        with gmm_dst.temporary_path() as tmp_path, \
                open(tmp_path, 'w') as out_file:
            json.dump(model, out_file, indent=2, sort_keys=True)


class FilterComponents(HelperTask):
    INPUT_DIR = HelperTask.INPUT_DIR

    datasets = luigi.ListParameter(description="Names of the datasets to use")

    def requires(self):
        return BuildGMM(datasets=self.datasets)
    
    def output(self):
        yield self._as_target('gmm_var_selection.csv')
        yield self._as_target('gmm_amp_selection.csv')
        yield self._as_target('gmm_final_selection.csv')

    def run(self):
        mu, sig, w, _ = self.input()
        mu = np.loadtxt(mu.path, delimiter=',')
        sig = np.loadtxt(sig.path, delimiter=',')
        w = np.loadtxt(w.path, delimiter=',')
        var_out, amp_out, final_out = self.output()

        var = sig ** 2
        var_99th_perc = matlab_alike_quantile(var, 0.99)
        var_inlier = var[var < var_99th_perc]
        var_thresholds = find_thresholds(var_inlier)
        var_selection = var < var_thresholds[-1]
        with var_out.temporary_path() as tmp_path:
            np.savetxt(tmp_path, var_selection.reshape(1, -1), delimiter=',')

        amp = np.array([
            # it doesn't matter where the actual mu is, we need max
            w_ * norm.pdf(0, 0, sig_) for w_, sig_
            in zip(w[var_selection], sig[var_selection])
        ])
        amp_inv = 1. / amp
        amp_inv_99th_perc = matlab_alike_quantile(amp_inv, 0.95)
        amp_inv_inlier = amp_inv[amp_inv < amp_inv_99th_perc]
        amp_inv_thresholds = find_thresholds(amp_inv_inlier)
        GAMRED_FILTER = 2
        amp_selection = amp_inv < amp_inv_thresholds[GAMRED_FILTER]
        with amp_out.temporary_path() as tmp_path:
            np.savetxt(tmp_path, amp_selection.reshape(1, -1), delimiter=',')
        
        final_selection = var_selection.copy()
        final_selection[final_selection] = amp_selection
        with final_out.temporary_path() as tmp_path:
            np.savetxt(tmp_path, final_selection.reshape(1, -1), delimiter=',')
