# -*- coding: utf-8 -*-
"""Resample mean spectrum

Created on Sat Jun  2 20:59:49 2018

Arguments:
    path to mz axis
    path to mean spectrum
    destination of mz axis
    destination of resampled spectrum

@author: Grzegorz Mrukwa
"""
import sys

import numpy as np

from components.spectrum.resampling import estimate_new_axis


resample = np.interp


EMPIRICAL_OPTIMAL_CHANNELS_NUMBER = 100000


if __name__ == '__main__':
    old_mz_axis = np.loadtxt(sys.argv[1])
    old_spectrum = np.loadtxt(sys.argv[2])
    axis_limits = np.min(old_mz_axis), np.max(old_mz_axis)
    new_mz_axis = estimate_new_axis(
            old_axis=old_mz_axis,
            number_of_ticks=EMPIRICAL_OPTIMAL_CHANNELS_NUMBER,
            axis_limits=axis_limits
    )
    np.savetxt(sys.argv[3], new_mz_axis)
    resampled_spectrum = resample(new_mz_axis, old_mz_axis, old_spectrum)
    np.savetxt(sys.argv[4], resampled_spectrum)
