import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.testing as npt
import scipy.stats

import spectrum.outlier


def returns(value):
    return MagicMock(return_value=value)


preselected_randoms = scipy.stats.zscore(np.arange(-1, 1.1, .1))


class TestDetect(unittest.TestCase):
    @patch.object(np.random, np.random.normal.__name__,
                  new=returns(preselected_randoms))
    def test_is_consistent_to_original(self):
        random_tics = np.array([1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2,
                                4, 5, 6, 7, 8, 10000, 1, 1, 1, 1, 1, 1, 1, 1, 2,
                                3, 2, 3, 2, 1, 2, 1, 1, 2, 3, 2, 1, 2, 3, 10,
                                0, 0, 0, 0, 0, 0, 0, 0, 0])
        outlier = spectrum.outlier.detect(random_tics)
        npt.assert_equal(outlier, random_tics == 10000)

    def test_finds_outlier(self):
        random_tics = np.array([1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2,
                                4, 5, 6, 7, 8, 1000, 1, 1, 1, 1, 1, 1, 1,
                                1, 2, 3, 2, 3, 2, 1, 2, 1, 1, 2, 3, 2, 1, 2,
                                3, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0] * 1000,
                               dtype=float)
        outlier = 2**64 - 1
        random_tics[-1] = outlier
        outliers = spectrum.outlier.detect(random_tics)
        npt.assert_equal(outliers, random_tics == outlier)
