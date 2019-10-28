import unittest

import numpy as np
import numpy.testing as npt

import components.spectrum.model as mdl


class TestMerge(unittest.TestCase):
    def setUp(self):
        self.means = np.arange(10.)
        self.sigmas = np.ones((10,))
        self.weights = np.ones((10,))

    def test_merges_none_components_if_too_distant(self):
        components = mdl.Components(self.means, .1 * self.sigmas, self.weights)
        merged = mdl.merge(components)
        self.assertEqual(len(merged.matches), len(components))

    def test_merges_at_most_4_components(self):
        self.sigmas[0] = 100.
        components = mdl.Components(self.means, .1 * self.sigmas, self.weights)
        merged = mdl.merge(components)
        self.assertEqual(len(merged.matches), len(components) - 3)

    def test_merges_each_components_within_4_sigma(self):
        components = mdl.Components(self.means, .5 * self.sigmas, self.weights)
        merged = mdl.merge(components)
        self.assertEqual(len(merged.matches), 4)

    def test_merged_components_have_proper_size_info(self):
        components = mdl.Components(self.means, .5 * self.sigmas, self.weights)
        merged = mdl.merge(components)
        self.assertEqual(merged.matches.lengths[0], 3)
        self.assertEqual(merged.matches.lengths[-1], 1)

    def test_merged_components_have_proper_start_index_info(self):
        components = mdl.Components(self.means, .5 * self.sigmas, self.weights)
        merged = mdl.merge(components)
        self.assertEqual(merged.matches.indices[0], 0)
        self.assertEqual(merged.matches.indices[-1], 9)

    def test_takes_mean_of_first_highest_component(self):
        self.weights[1] = 100.
        components = mdl.Components(self.means, .5 * self.sigmas, self.weights)
        merged = mdl.merge(components)
        self.assertEqual(merged.new_components.means[0], components.means[1])
        self.assertEqual(merged.new_components.means[1], components.means[3])


class TestApplyMerging(unittest.TestCase):
    def test_merges_data_according_to_matches(self):
        data = np.array([[.5, .5, .3, .3, .4]])
        matches = mdl.Matches([0, 2], [2, 3])
        merged = mdl.apply_merging(data, matches)
        self.assertEqual(merged.size, 2)
        self.assertAlmostEqual(merged[0][0], 1.)
        self.assertAlmostEqual(merged[0][1], 1.)

    def test_merges_multirow_data(self):
        data = np.array([
            [.5, .5, .3, 1.3, .4],
            [.4, .4, .3, 1.0, .4]
        ])
        matches = mdl.Matches([0, 2], [2, 3])
        merged = mdl.apply_merging(data, matches)
        expected = np.array([
            [1., 2.],
            [.8, 1.7]
        ], dtype=np.float32)
        self.assertEqual(merged.size, 4)
        npt.assert_array_equal(merged, expected)
