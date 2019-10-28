import unittest

import numpy as np

import experiment.convolve as conv


mzs = np.arange(100)


class TestRelevantRange(unittest.TestCase):
    def test_returns_two_numbers(self):
        left, right = conv.relevant_range(mzs, mu=20, sig=2)
        self.assertIsInstance(left, int)
        self.assertIsInstance(right, int)
