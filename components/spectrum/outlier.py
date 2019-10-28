from numba import jit
import numpy as np
from scipy.stats import norm

from components.seeding import seeded
from components.stats import \
    matlab_alike_quantile as quantile, matlab_alike_iqr as iqr

norminv = norm.ppf


def _median_iqr(data: np.ndarray):
    quant25, median_, quant75 = quantile(data, [.25, .5, .75])
    iqr_ = quant75 - quant25
    return median_, iqr_


def _nonparametric_normalize(data: np.ndarray, median_: float = None,
                             iqr_: float = None) -> np.ndarray:
    if median_ is None and iqr_ is None:
        median_, iqr_ = _median_iqr(data)
    elif median_ is None:
        median_ = np.median(data)
    elif iqr_ is None:
        iqr_ = iqr(data)
    return (data - median_) / iqr_


_magical_constant_for_w1 = 1.3426
_magical_constant_for_g1 = 1. / 1.29
_magical_constant_for_h1 = 1.29 ** 2


def _make_black_magic(normalized_tics, min_normalized_tic):
    # 2
    r = (normalized_tics - min_normalized_tic) + .1
    min_r, max_r = np.min(r), np.max(r)
    # 3
    r1 = r / (min_r + max_r)
    # 4
    w = norminv(r1, loc=0, scale=1)
    w_median, w_iqr = _median_iqr(w)
    # 5
    w1 = (w - w_median) / (w_iqr / _magical_constant_for_w1)
    # 6
    qp1, qp = quantile(w1, [.1, .9])
    g1 = _magical_constant_for_g1 * np.log(-qp / qp1)
    h1 = (2. * np.log(
        -g1 * ((qp * qp1) / (qp + qp1)))) / _magical_constant_for_h1
    return r, min_r, max_r, w, w_median, g1, h1


_empirically_selected_size = 100000


@jit(nopython=True)
def _make_y_tuk(g1, h1, Z):
    y_tuk = np.empty(Z.shape)
    g1_inv = 1. / g1
    for i in range(Z.size):
        y_tuk[i] = g1_inv * (np.exp(g1 * Z[i]) - 1) * (
            np.exp((h1 * (Z[i] ** 2)) / 2))
    return y_tuk


_magical_constant_for_f = 1. / 1.3426


def _f(w_median, w_iqr, L):
    return norm.cdf(w_median + w_iqr * _magical_constant_for_f * L,
                    loc=0, scale=1)


def _B(f, min_r, max_r, min_normalized_tic, tic_iqr, tic_median):
    return ((f * (min_r + max_r))
            + min_normalized_tic - 0.1) * tic_iqr + tic_median


@seeded()
def detect(tics):
    tic_median, tic_iqr = _median_iqr(tics)
    normalized_tics = _nonparametric_normalize(tics, tic_median, tic_iqr)
    min_normalized_tic = np.min(normalized_tics)
    r, min_r, max_r, w, w_median, g1, h1 = _make_black_magic(normalized_tics,
                                                             min_normalized_tic)
    Z = np.random.normal(loc=0, scale=1, size=_empirically_selected_size)
    y_tuk = _make_y_tuk(g1, h1, Z)
    # 7
    P = 1. - 1. / (2. * tics.size)
    # 8
    L1, L2 = quantile(y_tuk, [1. - P, P])
    w_iqr = iqr(w, rng=(10, 90))

    f1 = _f(w_median, w_iqr, L1)
    f2 = _f(w_median, w_iqr, L2)

    B1 = _B(f1, min_r, max_r, min_normalized_tic, tic_iqr, tic_median)
    B2 = _B(f2, min_r, max_r, min_normalized_tic, tic_iqr, tic_median)

    return np.logical_or(tics <= B1, tics >= B2)
