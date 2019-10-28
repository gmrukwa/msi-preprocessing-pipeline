from functools import partial
from typing import Tuple

from numba import jit
import numpy as np
import scipy.stats
import scipy.interpolate
from tqdm import tqdm


@jit(forceobj=True)
def _trend_exists(x: np.ndarray, y: np.ndarray, significance: float = .05) \
        -> bool:
    _, probability = scipy.stats.pearsonr(x, y)
    return probability < significance


_estimate_baseline = partial(np.percentile, q=10., interpolation='linear')


def do_nothing(*args, **kwargs): pass


@jit(forceobj=True)
def _estimates_and_widths(mzs: np.ndarray, counts: np.ndarray,
                          max_width: int = 1500,
                          minimal_width: int = 500,
                          width_increment: int = 100,
                          verbose: bool = False) \
        -> Tuple[np.ndarray, np.ndarray]:
    if verbose:
        progress_bar = tqdm(total=mzs.size)
        report_progress = progress_bar.update
    else:
        report_progress = do_nothing
    start, end = 0, min(minimal_width, mzs.size)
    baseline_estimates, segment_widths = [], []
    while end <= mzs.size:
        segment, segment_mzs = counts.ravel()[start:end], mzs.ravel()[start:end]
        if _trend_exists(segment_mzs, segment) or segment.size >= max_width:
            baseline_estimates.append(_estimate_baseline(segment))
            segment_widths.append(segment.size)
            start, end = end, end + minimal_width
            report_progress(segment.size)
        else:
            end = min(end + width_increment, mzs.size + 1)
    if start < mzs.size:
        baseline_estimates.append(_estimate_baseline(counts.ravel()[start:]))
        segment_widths.append(counts[start:].size)
    if verbose:
        progress_bar.close()
    return np.array(baseline_estimates), np.array(segment_widths)


def _segment_ends_from_widths(widths):
    return np.cumsum(widths) - 1


def _mzs_estimate_by_ends(ends, mzs):
    estimated_mzs = mzs[((ends[2:] + ends[1:-1]) * .5).astype(int)]
    estimated_mzs = np.hstack((
        [mzs[int((ends[0] - 1) * .5)]],
        estimated_mzs,
        [(mzs[ends[-1]] + mzs[-1]) * .5]
    ))
    return estimated_mzs


_cubic_model = partial(scipy.interpolate.interp1d, kind='cubic',
                       fill_value='extrapolate')


def _remove_model(mzs, counts, model):
    return np.clip(counts - model(mzs), a_min=0., a_max=None)


def adaptive_remove(mzs: np.ndarray, counts: np.ndarray,
                    max_segment_width: int = 1500,
                    minimal_segment_width: int = 500,
                    width_increment: int = 100,
                    verbose: bool = False) -> np.ndarray:
    baseline_estimates, segment_widths = _estimates_and_widths(
        mzs, counts, max_segment_width, minimal_segment_width, width_increment,
        verbose)
    baseline_segment_ends = _segment_ends_from_widths(segment_widths)
    baseline_mzs_estimate = _mzs_estimate_by_ends(baseline_segment_ends, mzs)
    baseline_model = _cubic_model(baseline_mzs_estimate, baseline_estimates)
    clean_counts = _remove_model(mzs, counts, baseline_model)
    return clean_counts.astype(np.float32)
