import unittest

import numpy as np
import numpy.testing as npt

import components.stats as stats


class TestMatlabAlikeQuantiles(unittest.TestCase):
    def test_is_consistent_with_original(self):
        values = np.arange(11)
        quantiles = np.arange(0, 1.1, .1)
        quantile_values = stats.matlab_alike_quantile(values, quantiles)
        expected = np.array([0, .6, 1.7, 2.8, 3.9, 5., 6.1, 7.2, 8.3, 9.4, 10.])
        npt.assert_almost_equal(quantile_values, expected)
