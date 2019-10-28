import unittest
from unittest.mock import patch

import numpy as np
import numpy.testing as npt

import spectrum.baseline as ba


class TestTrendExists(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_finds_trend_in_simple_data(self):
        self.assertTrue(ba._trend_exists([1, 2, 3], [2, 3, 4]))

    def test_finds_trend_in_noise_with_high_significance(self):
        self.assertTrue(ba._trend_exists(
            np.arange(100), np.random.rand(100), significance=1.))

    def test_returns_false_for_random_data(self):
        self.assertFalse(ba._trend_exists(
            np.arange(100), np.random.rand(100), significance=.01))


def simulate_mzs(n):
    return np.sort(.04 * np.arange(n) ** 2 + .2 * np.arange(n) +
                   np.random.randn(n))


class TestEstimatesAndWidths(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_estimates_and_widths_are_equal_size(self):
        mzs = simulate_mzs(100)
        counts = np.random.randint(0, 31, 100)
        estimates, widths = ba._estimates_and_widths(mzs, counts)
        self.assertEqual(estimates.size, widths.size)

    # # This one is disabled due to wrong quantiles calculated from MATLAB
    # # the original code was in.
    # def test_is_consistent_with_original(self):
    #     mzs = np.arange(100)
    #     signal = 0.1 * mzs + np.sin(mzs)
    #     expected_widths = [15, 13, 12, 13, 13, 6, 12, 13, 3]
    #     expected_estimates = [-0.356802495307928, 0.986930704206748,
    #                           2.16904679382306, 3.43127589631399,
    #                           4.77301772624698, 5.85023925085328,
    #                           6.56071575203841, 7.83785404129453,
    #                           8.90079316581365]
    #     estimates, widths = ba._estimates_and_widths(mzs, signal, 15, 5, 1)
    #     npt.assert_almost_equal(expected_estimates, estimates)
    #     npt.assert_almost_equal(expected_widths, widths)


# Following tests are temporarily not implemented since it is all tested in
# integration. These are to be implemented as soon as there is a time for this.
# Right now the only difference is due to valid quantiles computation.
class TestSegmentEndsFromWidths(unittest.TestCase):
    def test_finds_ends_for_each_segment(self):
        pass


# Following tests are temporarily not implemented since it is all tested in
# integration. These are to be implemented as soon as there is a time for this.
# Right now the only difference is due to valid quantiles computation.
class TestMzsEstimateByEnds(unittest.TestCase):
    def test_finds_mzs_for_each_segment(self):
        pass

    def test_has_artificial_last_mz(self):
        pass


# Following tests are temporarily not implemented since it is all tested in
# integration. These are to be implemented as soon as there is a time for this.
# Right now the only difference is due to valid quantiles computation.
class TestRemoveModel(unittest.TestCase):
    def test_moves_plot_lower_by_model(self):
        pass

    def test_clips_minimum_to_zero(self):
        pass


class TestAdaptiveRemove(unittest.TestCase):
    def test_is_consistent_with_original(self):
        mzs = np.arange(100)
        signal = 0.1 * mzs + np.sin(mzs)
        expected_removal_effect = np.array(
            [0.0172330735022189, 1.03843080781518, 1.27638044748339,
             0.668985784643117, 0, 0, 0.677386997109002, 1.73984118435010,
             2.19024003150190, 1.72326568842104, 0.869892839773969,
             0.509454930909737, 1.06143095569989, 2.10002030633938,
             2.74586379072228, 2.47476432053121, 1.59987319901746,
             0.984022159202895, 1.24668174993193, 2.19466487233869,
             2.99998400985622, 2.96134103472795, 2.14913937470500,
             1.34099732849368, 1.30705129033536, 2.10213780304676,
             3.01561899568586, 3.22498166706139, 2.55229403193403,
             1.62803728603086, 1.31168600454677, 1.90175308362024,
             2.86158026252629, 3.31298115590052, 2.84388366871903,
             1.88742908233544, 1.32398586017765, 1.67198484473927,
             2.61151823409722, 3.27870324190929, 3.06017384941748,
             2.15724859665163, 1.40108114739395, 1.48874334599901,
             2.34258248064556, 3.18185673071105, 3.24064343123019,
             2.47170208519669, 1.58991888907519, 1.41463490227570,
             2.11579483026815, 3.05714782066460, 3.38066073723473,
             2.79483704288462, 1.84216454610688, 1.39980178621441,
             1.87256969207112, 2.82020826157296, 3.36159675177054,
             2.98453125932765, 2.01699933002330, 1.32544748811769,
             1.51866953417008, 2.38881162713707, 3.10319944199684,
             2.97062239883583, 2.07755677707643, 1.20938711771781,
             1.12905459306373, 1.87630699948265, 2.73175392120207,
             2.87894602391169, 2.15559438683657, 1.20332506077781,
             0.878317908730098, 1.46468566103407, 2.41380885406804,
             2.84928090772745, 2.37305475511788, 1.43130796111752,
             0.904513438998674, 1.29774104583823, 2.27593860552130,
             2.97161725464010, 2.78205662323238, 1.92308308368840,
             1.23027989966084, 1.39039574730330, 2.30959106986959,
             3.19935365377404, 3.29901703573952, 2.56661697329337,
             1.71379776246624, 1.54179365839925, 2.19296569474674,
             3.00810350436447, 3.12068818925615, 2.24175369773299,
             0.913748778199656, 0])
        faulty_original_quantiles = [-0.356802495307928, 0.986930704206748,
                                     2.16904679382306, 3.43127589631399,
                                     4.77301772624698, 5.85023925085328,
                                     6.56071575203841, 7.83785404129453,
                                     8.90079316581365]
        with patch.object(ba, '_estimate_baseline',
                          side_effect=faulty_original_quantiles):
            removed = ba.adaptive_remove(mzs, signal, 15, 5, 1)
        npt.assert_almost_equal(removed, expected_removal_effect)


if __name__ == '__main__':
    unittest.main()
