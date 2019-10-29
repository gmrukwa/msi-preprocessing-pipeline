import unittest

import numpy as np
from numpy import testing as npt

import components.spectrum.resampling as resampling


class TestCenterizedScaledDomain(unittest.TestCase):
    def test_has_required_number_of_elements(self):
        domain = resampling._centered_scaled_domain(10)
        self.assertEqual(domain.size, 10)

    def test_is_centerized_to_zero(self):
        domain = resampling._centered_scaled_domain(10)
        self.assertAlmostEqual(domain.mean(), 0., places=3)

    def test_is_scaled_to_1(self):
        domain = resampling._centered_scaled_domain(10)
        self.assertAlmostEqual(domain.max(), 1., places=3)


class TestEstimateNewAxis(unittest.TestCase):
    def test_modelled_axis(self):
        old_axis = np.arange(101) ** 2
        narrow_range = np.array([1, 20])
        number_of_samples = 10
        expected_output = np.array([1, 1.0211, 1.5647, 2.6308, 4.2194,
                                    6.3306, 8.9642, 12.1203, 15.7989, 20])
        new_axis = resampling.estimate_new_axis(old_axis, number_of_samples, narrow_range)
        npt.assert_almost_equal(new_axis, expected_output, decimal=4)

    def test_large_modelled_axis(self):
        old_axis = np.arange(1000001) ** 2
        narrow_range = np.array([0, 1000000])
        number_of_samples = 100
        expected_output = np.array([
            0, 0.01010, 206.16305, 618.45887, 1236.89754, 2061.47907,
            3092.20346, 4329.07070, 5772.08080, 7421.23376, 9276.52958,
            11337.96825, 13605.54978, 16079.27417, 18759.14141, 21645.15151,
            24737.30447, 28035.60028, 31540.03896, 35250.62049, 39167.34487,
            43290.21212, 47619.22222, 52154.37518, 56895.67099, 61843.10966,
            66996.69119, 72356.41558, 77922.28282, 83694.29292, 89672.44588,
            95856.74170, 102247.18037, 108843.76190, 115646.48629, 122655.35353,
            129870.36363, 137291.51659, 144918.81240, 152752.25108,
            160791.83261, 169037.55699, 177489.42424, 186147.43434,
            195011.58730, 204081.88311, 213358.32178, 222840.90331,
            232529.62770, 242424.49494, 252525.50505, 262832.65800,
            273345.95382, 284065.39249, 294990.97402, 306122.69841,
            317460.56565, 329004.57575, 340754.72871, 352711.02453,
            364873.46320, 377242.04473, 389816.76911, 402597.63636,
            415584.64646, 428777.79942, 442177.09523, 455782.53391,
            469594.11544, 483611.83982, 497835.70707, 512265.71717,
            526901.87012, 541744.16594, 556792.60461, 572047.18614,
            587507.91053, 603174.77777, 619047.78787, 635126.94083,
            651412.23665, 667903.67532, 684601.25685, 701504.98124,
            718614.84848, 735930.85858, 753453.01154, 771181.30735,
            789115.74603, 807256.32756, 825603.05194, 844155.91919,
            862914.92929, 881880.08225, 901051.37806, 920428.81673,
            940012.39826, 959802.12265, 979797.98989, 1000000
        ])
        new_axis = resampling.estimate_new_axis(old_axis, number_of_samples,
                                                         narrow_range)
        npt.assert_almost_equal(new_axis, expected_output, decimal=4)


if __name__ == '__main__':
    unittest.main()
