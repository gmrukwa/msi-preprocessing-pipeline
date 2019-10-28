from functools import lru_cache

import numpy as np


@lru_cache()
def _centered_scaled_domain(number_of_elements):
    centered = np.arange(1. - number_of_elements, number_of_elements + 1., 2)
    assert centered.size == number_of_elements, \
        str(centered.size) + '!=' + str(number_of_elements)
    return centered / (number_of_elements - 1)


tick_distances = np.diff


def _exclude_outliers(ticks, preserved: np.ndarray=None):
    if preserved is None:
        preserved = np.ones((ticks.size,), dtype=np.bool)
    domain = _centered_scaled_domain(ticks.size)
    coefficients = np.polyfit(domain[preserved], ticks[preserved], 1)
    residuals = (np.polyval(coefficients, domain) - ticks) ** 2
    preserved = residuals <= 2 * np.mean(residuals)
    return coefficients, preserved


def _ticks_are_equalized(tick_estimate):
    return np.abs(np.mean(tick_estimate) / np.diff(tick_estimate)) > 1000


def _equally_spaced_ticks(number_of_ticks, axis_limits):
    return axis_limits[0] + np.arange(number_of_ticks) \
        * np.diff(axis_limits)/number_of_ticks


def _modelled_ticks(tick_estimate, number_of_ticks, axis_limits):
    first_scale = np.diff(axis_limits) * 2. / np.sum(tick_estimate) / (number_of_ticks - 1)
    second_scale = np.diff(tick_estimate) * first_scale / (number_of_ticks - 2)
    base = np.arange(number_of_ticks-1)
    const_increment = ((base - 1.) * base / 2.) * second_scale
    modelled_increment = base * (first_scale * tick_estimate[0])
    return np.hstack([
            axis_limits[0] + modelled_increment + const_increment,
            [axis_limits[1]]
    ])


def estimate_new_axis(old_axis, number_of_ticks, axis_limits):
    ticks = tick_distances(old_axis)
    coefficients, preserved = _exclude_outliers(ticks)
    previous_allowed = np.inf
    for _ in range(9):
        if np.sum(preserved) == previous_allowed:
            break
        previous_allowed = np.sum(preserved)
        coefficients, preserved = _exclude_outliers(ticks, preserved)
    domain = _centered_scaled_domain(ticks.size)
    tick_estimate = np.polyval(coefficients, [domain[0], domain[-1]])
    if _ticks_are_equalized(tick_estimate):
        return _equally_spaced_ticks(number_of_ticks, axis_limits)
    else:
        return _modelled_ticks(tick_estimate, number_of_ticks, axis_limits)
