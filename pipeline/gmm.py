import json
import os
from functools import partial

import luigi
import numpy as np
from scipy.stats import norm

import components.spectrum.model as mdl
from components.convolve import convolve
from components.matlab_legacy import estimate_gmm, find_thresholds
from components.spectrum.resampling import estimate_new_axis
from components.stats import matlab_alike_quantile
from pipeline._base import *
from pipeline.normalize import NormalizeTIC
from pipeline.outlier import DetectOutliers
from pipeline.resampling import FindResamplingAxis
from plot import save_decomposition


save_csv = partial(np.savetxt, delimiter=',')
load_csv = partial(np.loadtxt, delimiter=',')


def save_csv_tmp(out: luigi.LocalTarget, X, *args, **kwargs):
    with out.temporary_path() as tmp_path:
        save_csv(tmp_path, X, *args, **kwargs)


class ExtractGMMReference(ExtractReference):
    INPUT_DIR = NormalizeTIC.OUTPUT_DIR

    datasets = luigi.ListParameter(description="Names of the datasets to use")

    def requires(self):
        yield DetectOutliers(datasets=self.datasets)
        for dataset in self.datasets:
            yield NormalizeTIC(dataset=dataset, datasets=self.datasets)
    
    def output(self):
        return self._as_target("gmm_reference.csv")


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
        self.set_status_message('Loading data')
        old_mzs, old_reference = self.input()
        reference_dst, new_mzs_dst = self.output()
        old_mzs = load_csv(old_mzs.path)
        old_reference = load_csv(old_reference.path)

        self.set_status_message('Estimating new m/z axis')
        limits = np.min(old_mzs), np.max(old_mzs)
        new_mzs = estimate_new_axis(
            old_axis=old_mzs,
            number_of_ticks=EMPIRICAL_OPTIMAL_CHANNELS_NUMBER,
            axis_limits=limits
        )
        save_csv_tmp(new_mzs_dst, new_mzs.reshape(1, -1))
        
        self.set_status_message('Resampling the reference spectrum')
        resampled = resample(new_mzs, old_mzs, old_reference)
        save_csv_tmp(reference_dst, resampled.reshape(1, -1))


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
        self.set_status_message('Loading data')
        spectrum, mzs = self.input()
        mu_dst, sig_dst, w_dst, gmm_dst = self.output()
        spectrum = load_csv(spectrum.path)
        mzs = load_csv(mzs.path)
        
        self.set_status_message('Estimating GMM model')
        mu, sig, w, model = estimate_gmm(mzs, spectrum)

        logger.info('Found {0} GMM components'.format(mu.size))
        self.set_status_message('Saving {0} GMM components'.format(mu.size))
        save_csv_tmp(mu_dst, mu)
        save_csv_tmp(sig_dst, sig)
        save_csv_tmp(w_dst, w)
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
        yield self._as_target('filtered_mu.csv')
        yield self._as_target('filtered_sig.csv')
        yield self._as_target('filtered_w.csv')
        yield self._as_target('thresholds_amp.html')
        yield self._as_target('thresholds_var.html')

    def run(self):
        self.set_status_message('Loading data')
        mu, sig, w, _ = self.input()
        mu = load_csv(mu.path)
        sig = load_csv(sig.path)
        w = load_csv(w.path, delimiter=',')
        var_out, amp_out, final_out, filt_mu, filt_sig, filt_w, \
            amp_plot, var_plot = self.output()

        msg = 'Amplitude filtering (out of {0})'.format(mu.size)
        logger.info(msg)
        self.set_status_message(msg)
        amp = np.array([
            # it doesn't matter where the actual mu is, we need max
            w_ * norm.pdf(0, 0, sig_) for w_, sig_
            in zip(w, sig)
        ])
        amp_inv = 1. / amp
        amp_inv_95th_perc = matlab_alike_quantile(amp_inv, 0.95)
        amp_inv_inlier = amp_inv[amp_inv < amp_inv_95th_perc]
        amp_inv_thresholds = find_thresholds(amp_inv_inlier)
        GAMRED_FILTER = 2
        amp_selection = amp_inv < amp_inv_thresholds[GAMRED_FILTER]
        save_decomposition(amp_inv_inlier, amp_inv_thresholds, amp_plot)
        save_csv_tmp(amp_out, amp_selection.reshape(1, -1), fmt='%i')

        msg = 'Variance filtering (out of {0})'.format(np.sum(amp_selection))
        logger.info(msg)
        self.set_status_message(msg)
        var = sig[amp_selection] ** 2
        var_95th_perc = matlab_alike_quantile(var, 0.95)
        var_inlier = var[var < var_95th_perc]
        var_thresholds = find_thresholds(var_inlier)
        logger.info("Found {0} thresholds".format(len(var_thresholds)))
        var_selection = np.ones_like(var, dtype=bool)
        for idx, thr in enumerate(var_thresholds[::-1]):
            var_selection = var < thr
            if 1000 <= np.sum(var_selection) <= 3500:
                logger.info("Selected {0} highest threshold".format(idx))
                break
            else:
                logger.info("Dropped threshold allowing for {0} components"
                            .format(np.sum(var_selection)))
        else:
            logger.info("Selected just some threshold (not a preferred).")
        save_decomposition(var_inlier, var_thresholds, var_plot)
        save_csv_tmp(var_out, var_selection.reshape(1, -1), fmt='%i')
        
        final_selection = amp_selection.copy()
        final_selection[final_selection] = var_selection
        
        msg = 'Saving {0} filtered components'.format(np.sum(final_selection))
        logger.info(msg)
        self.set_status_message(msg)
        save_csv_tmp(final_out, final_selection.reshape(1, -1), fmt='%i')
        save_csv_tmp(filt_mu, mu[final_selection].reshape(1, -1))
        save_csv_tmp(filt_sig, sig[final_selection].reshape(1, -1))
        save_csv_tmp(filt_w, w[final_selection].reshape(1, -1))


class Convolve(BaseTask):
    INPUT_DIR = NormalizeTIC.OUTPUT_DIR
    OUTPUT_DIR = os.path.join(BaseTask.OUTPUT_DIR, '06-gmm-convolved')

    dataset = luigi.Parameter(description="Dataset to convolve")
    datasets = luigi.ListParameter(
        description="Names of the datasets to use",
        visibility=luigi.parameter.ParameterVisibility.HIDDEN)

    def requires(self):
        yield FilterComponents(datasets=self.datasets)
        yield FindResamplingAxis(datasets=self.datasets)
        yield NormalizeTIC(dataset=self.dataset, datasets=self.datasets)
    
    def output(self):
        return self._as_target("{0}.npy".format(self.dataset))
    
    def run(self):
        self.set_status_message('Loading data')
        gmm, mzs, spectra = self.input()
        *_, mu, sig, w, _, _ = gmm
        mzs = load_csv(mzs.path).ravel()
        mu = load_csv(mu.path).ravel()
        sig = load_csv(sig.path).ravel()
        w = load_csv(w.path).ravel()
        spectra = np.load(spectra.path)

        self.set_status_message('Convolving')
        convolved = convolve(spectra, mzs, mu, sig, w)

        self.set_status_message('Saving results')
        with self.output().temporary_path() as tmp_path, \
                open(tmp_path, 'wb') as out_file:
            np.save(out_file, convolved)


class MergeComponents(HelperTask):
    INPUT_DIR = HelperTask.OUTPUT_DIR

    datasets = luigi.ListParameter(description="Names of the datasets to use")

    def requires(self):
        return FilterComponents(datasets=self.datasets)
    
    def output(self):
        yield self._as_target('merged_start_indices.csv')
        yield self._as_target('merged_lengths.csv')
        yield self._as_target('merged_mu.csv')
        yield self._as_target('merged_sig.csv')
        yield self._as_target('merged_w.csv')
    
    def run(self):
        self.set_status_message('Loading data')
        *_, mu, sig, w, _, _ = self.input()
        mu = load_csv(mu.path).ravel()
        sig = load_csv(sig.path).ravel()
        w = load_csv(w.path).ravel()

        self.set_status_message('Components merging')
        merged = mdl.merge(mdl.Components(mu, sig, w))
        msg = "{0} merged components, compression rate {1}".format(
            merged.matches.indices.size,
            merged.matches.lengths.mean())
        logger.info(msg)
        self.set_status_message(msg)
        
        indices, lengths, mu_dst, sig_dst, w_dst = self.output()
        save_csv_tmp(indices, merged.matches.indices, fmt='%i')
        save_csv_tmp(lengths, merged.matches.lengths, fmt='%i')
        save_csv_tmp(mu_dst, merged.new_components.means)
        save_csv_tmp(sig_dst, merged.new_components.sigmas)
        save_csv_tmp(w_dst, merged.new_components.weights)


class MergeDataset(BaseTask):
    INPUT_DIR = Convolve.OUTPUT_DIR
    OUTPUT_DIR = os.path.join(BaseTask.OUTPUT_DIR, '07-gmm-merged')

    dataset = luigi.Parameter(description="Dataset to convolve")
    datasets = luigi.ListParameter(
        description="Names of the datasets to use",
        visibility=luigi.parameter.ParameterVisibility.HIDDEN)

    def requires(self):
        yield Convolve(dataset=self.dataset, datasets=self.datasets)
        yield MergeComponents(datasets=self.datasets)
    
    def output(self):
        yield self._as_target('{0}.npy'.format(self.dataset))
        yield self._as_target('mz.csv')
    
    def run(self):
        self.set_status_message('Loading data')
        spectra, components = self.input()
        indices, lengths, mu, *_ = components
        indices = load_csv(indices.path, dtype=int)
        lengths = load_csv(lengths.path, dtype=int)
        mu = load_csv(mu.path).reshape(1, -1)
        spectra = np.load(spectra.path)

        self.set_status_message('Merging components')
        merged = mdl.apply_merging(spectra, mdl.Matches(indices, lengths))

        self.set_status_message('Saving results')
        spectra_dst, mz_dst = self.output()
        if not mz_dst.exists():
            save_csv_tmp(mz_dst, mu)
        with spectra_dst.temporary_path() as tmp_path, \
                open(tmp_path, 'wb') as out_file:
            np.save(out_file, merged)
