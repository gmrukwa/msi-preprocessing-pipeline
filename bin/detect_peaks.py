"""Detect peaks

Find relevant peaks on smoothed mean spectrum

Arguments:
    mean spectrum path
    mz axis path
    smoothing window size (int)
    peak list destination path
    mz list destination path
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

import components.spectrum.smoothing
import components.spectrum.peak


if __name__ == '__main__':
    mean_spectrum = np.loadtxt(sys.argv[1])
    mz = np.loadtxt(sys.argv[2])
    smooth = spectrum.smoothing.savitzky_golay(mean_spectrum,
                                               window=int(sys.argv[3]))
    gmm_reference = spectrum.peak.Spectrum(mz, smooth)
    peak_list, _ = spectrum.peak.gradient_detection(gmm_reference)
    np.savetxt(sys.argv[4], peak_list, fmt='%i')
    np.savetxt(sys.argv[5], mz[(peak_list,)])
    # @gmrukwa: sometimes it is good to see, what's inside
    # plt.plot(
    #     mz, mean_spectrum, 'g-',
    #     mz, smooth, 'b-',
    #     mz[(peak_list,)], smooth[(peak_list,)], 'r.'
    # )
    # plt.show()
