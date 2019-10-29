import unittest

import numpy as np
import numpy.testing as npt

import components.spectrum.alignment as al


def simulate_mzs(n):
    return np.sort(.04 * np.arange(n) ** 2 + .2 * np.arange(n) + \
                   np.random.randn(n))


mzs = np.array([0.64015721, 1.02272212, 1.53873798, 1.76405235,
                3.2008932, 3.20864279, 3.30755799, 3.59008842,
                4.05678115, 5.4505985, 6.14404357, 8.49427351,
                8.92103773, 9.48167502, 11.08386323, 12.33367433,
                14.75484174, 14.93407907, 16.8730677, 17.38590426,
                17.44701018, 22.4936186, 24.6244362, 25.01783498,
                28.54563433, 30.10975462, 32.28575852, 34.37281615,
                38.49277921, 40.90935877, 42.15494743, 45.01816252,
                46.47221425, 48.17920353, 52.69208785, 56.15634897,
                60.27029068, 63.36237985, 64.97267318, 68.33769725,
                70.95144703, 74.01998206, 77.25372981, 84.5107754,
                85.73034782, 89.5619257, 92.58720464, 98.53749036,
                100.14610215, 105.62725972, 109.10453344, 114.6269025,
                118.04919486, 121.77936782, 127.41181777, 132.42833187,
                136.70651722, 141.6624719, 145.52567791, 150.67725883,
                155.32753955, 160.68044684, 165.34685372, 169.6337174,
                176.81742614, 181.59821906, 185.80980165, 193.42278226,
                197.65270164, 204.2919454, 210.72909056, 215.96898291,
                222.89940068, 226.52517418, 234.24234164, 239.31518991,
                245.36920285, 251.98115034, 258.64844747, 265.49616534,
                270.83485016, 279.54082649, 285.82566244, 290.62375631,
                300.52825219, 307.89588918, 314.21877957, 319.98007516,
                326.28924738, 335.69445173, 341.59682305, 350.66244507,
                357.16827498, 365.53663904, 372.5963664, 380.70657317,
                387.85050002, 397.54587049, 403.88691209, 412.24198936])


class TestSegmentMaxSize(unittest.TestCase):
    def test_is_integer(self):
        size = al._segment_max_size(np.array([1, 2, 3]))
        self.assertIsInstance(size, int)

    def test_is_non_negative(self):
        size = al._segment_max_size(np.array([1, 2, 3]))
        self.assertGreaterEqual(size, 0)

    def test_is_around_twentieth(self):
        size = al._segment_max_size(np.arange(1000))
        self.assertAlmostEqual(50, size)
        size = al._segment_max_size(np.array([1, 2, 3]))
        self.assertEqual(size, 0)


class TestFindSegmentCutPoint(unittest.TestCase):
    def test_finds_without_matching_points(self):
        first = np.sin(mzs[:50])
        second = np.cos(mzs[:50] + 15)
        position = al._find_segment_cut_point(first, second)
        self.assertEqual(position, 15)

    def test_finds_matching_points(self):
        first = np.sin(mzs[:50])
        second = -np.cos(mzs[:50] + 1.5)
        position = al._find_segment_cut_point(first, second)
        self.assertEqual(position, 15)

    def test_is_consistent_with_original(self):
        first = np.array([
            -91.88295583342023, -99.4265499424876, -98.5879227989812,
            -48.15917206662814, -48.669727046205466, -11.465355625907149,
            -26.79845399993735, -96.5190802031737, 76.42205833859722,
            18.365463219448888, 71.29297517456608, -6.860035332764011])
        second = np.array([
            89.73180081626778, 56.531790551406516, 51.38895323127968,
            97.89604564034974, -34.714778457509865, -68.01203659785304,
            90.61680571909183, 42.895791814503035, -98.6927839729902,
            62.733787952404555, -99.63296507800811, 80.25082315613517])
        position = al._find_segment_cut_point(first, second)
        self.assertEqual(position, 2)


class TestShiftSegment(unittest.TestCase):
    def setUp(self):
        self.matrix = np.array([1, 2, 3, 4])

    def test_makes_no_shift_with_zero(self):
        npt.assert_almost_equal(al._shift_segment(self.matrix, 0), self.matrix)

    def test_fills_up_left_when_positive_shift(self):
        npt.assert_almost_equal(al._shift_segment(self.matrix, 1),
                                np.hstack([[1], self.matrix[:-1]]))

    def test_fills_up_right_when_negative_shift(self):
        npt.assert_almost_equal(al._shift_segment(self.matrix, -1),
                                np.hstack([self.matrix[1:], [4]]))

    def test_preserves_size(self):
        shifted = al._shift_segment(self.matrix, 1)
        self.assertEqual(self.matrix.size, shifted.size)
        shifted = al._shift_segment(self.matrix, -1)
        self.assertEqual(self.matrix.size, shifted.size)


class TestEstimatePadding(unittest.TestCase):
    def test_bit_width_based_padding(self):
        self.assertEqual(al._estimate_padding(10), 16)

    def test_custom_padding_for_long_cases(self):
        self.assertEqual(al._estimate_padding(1234567), 2234567)


class TestConvolveWithReversedTime(unittest.TestCase):
    def test_computes_convolution(self):
        first = np.sin(mzs[:50])
        second = np.cos(mzs[:50] + 15)
        size = al._estimate_padding(first.size)
        convolution = al._convolve_with_reversed_time(first, second, size)
        expected = np.array([
            -0.246424823939000, 0.0753453021683723, 0.0529355260026504,
            -0.0198059528286075, 0.0912815213621514, -0.00357081516833846,
            0.0123261926004330, 0.0595196762008749, -0.00516532266067159,
            0.0127462260460925, -0.00536594114832110, -0.0952991003833967,
            0.0332264576210169, 0.0256202280491808, -0.0561859899364777,
            0.0317047507066096, 0.0292668052710699, 0.0765614406647998,
            0.117552432911069, 0.0234972647840388, 0.00849985395325384,
            -0.0267190990043384, -0.0835904394277075, -0.0288988459095760,
            -0.0611317522537336, -0.0559264531436517, -0.0315400184546741,
            -0.00248298437990327, -0.0137194107476360, -0.0249465457691262,
            0.0381667578291461, 0.0316607225388498, 0.0204698820874805,
            0.00450522451074661, -0.0167075593173271, -0.0221328072906556,
            -0.0699503867362448, -0.0471164808747370, -0.0263897417499238,
            -0.0348625172627407, 0.0617535605069357, 0.0253405269149494,
            -2.14316242373863e-05, 0.0864631999828358, 0.0615505307122036,
            0.0358200795082400, 0.104283158413872, -0.0722695121451226,
            -0.0418188554917807, -0.0252325320494851, -0.0932333565991769,
            0.0524188416837890, -0.0547893518353792, -0.0524261163627246,
            0.0357146624821272, 0.0261328053695618, 0.0120841254743653,
            0.0731404674238546, -0.0639127747424580, -0.0333543138883225,
            0.0820320343716254, -0.131354741421251, -0.0379754792599599,
            -0.120473101755154])
        npt.assert_almost_equal(convolution, expected, 4)


class TestGetPeakWithinShiftLimit(unittest.TestCase):
    def test_returns_zero_shift_when_too_limited(self):
        shift = al._get_peak_within_shift_limit(np.arange(10), 0)
        self.assertEqual(shift, 0)

    def test_returns_zero_shift_when_peak_too_low(self):
        shift = al._get_peak_within_shift_limit(.01 * np.arange(10), 100)
        self.assertEqual(shift, 0)

    def test_returns_nonnegative_shift_for_first_half(self):
        shift = al._get_peak_within_shift_limit(10 - np.arange(10), 3)
        self.assertGreaterEqual(shift, 0)

    def test_returns_nonpositive_shift_for_second_half(self):
        shift = al._get_peak_within_shift_limit(np.arange(10), 3)
        self.assertLessEqual(shift, 0)

    def test_is_consistent_with_original(self):
        get_shift = al._get_peak_within_shift_limit
        self.assertEqual(-1, get_shift(np.arange(10), 3))
        self.assertEqual(0, get_shift(10 - np.arange(10), 3))
        self.assertEqual(8, get_shift(np.sin(np.arange(100)), 10))
        self.assertEqual(-9, get_shift(1. - np.cos(np.arange(100)), 10))


class TestFindShift(unittest.TestCase):
    def test_finds_shift_within_limit(self):
        counts = 100 * np.sin(mzs[:50])
        reference = 100 * np.cos(mzs[:50] + 15)
        shift = al._find_shift(counts, reference, 10)
        self.assertLessEqual(shift, 10)

    def test_limits_shift_as_requested(self):
        counts = 100 * np.sin(mzs[:50])
        reference = 100 * np.cos(mzs[:50] + 15)
        shift = al._find_shift(counts, reference, 3)
        self.assertLessEqual(shift, 3)

    def test_consistency_with_original(self):
        first = np.array([
            -6.699990961860916, -16.52044800133434, -43.36105575129196,
            -79.26775150349647, -73.96746723611477, -13.869319663041018,
            80.19214185033786, 48.27045430858234, -5.686636212317998,
            -99.61050621323385, -23.06019756890941, 81.52272811113734,
            69.8918485220403, -91.88295583342023, -99.4265499424876,
            -98.5879227989812])
        second = np.array([
            80.15501065882263, 85.6679761741285, 96.65267354793275,
            97.8605577845761, -3.0241642089437373, -66.215533098087,
            -6.761975752143003, 35.14250645724448, 79.54380890294239,
            58.07700680449529, -58.92552678575663, -9.016585079645171,
            8.88316180203674, 89.73180081626778, 56.531790551406516,
            51.38895323127968])
        expected_shift = -4
        shift_limit = 38
        shift = al._find_shift(first, second, shift_limit)
        self.assertEqual(expected_shift, shift)


class TestPafftIntegration(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_aligns_signal_without_shifting(self):
        counts = np.array([32, 36, 6, 21, 31, 13, 7, 24, 15, 18, 15, 11, 38, 29,
                           1, 31, 24, 24, 3, 18, 3, 12, 38, 35, 22, 5, 23, 32,
                           11, 20, 10, 37, 28, 2, 27, 19, 25, 23, 20, 29, 3, 35,
                           39, 9, 9, 23, 3, 26, 3, 31, 9, 10, 27, 7, 39, 21, 33,
                           34, 34, 24, 33, 5, 36, 0, 11, 34, 5, 16, 8, 1, 17,
                           35, 27, 36, 25, 3, 39, 35, 30, 29, 33, 18, 17, 29,
                           20, 2, 5, 37, 12, 2, 27, 21, 39, 39, 11, 22, 30,
                           17, 6, 7])
        reference_counts = np.array([18, 28, 19, 29, 21, 9, 25, 32, 27, 9, 28,
                                     17, 0, 22, 16, 36, 30, 24, 3, 8, 27, 29,
                                     23, 32, 19, 8, 7, 23, 13, 17, 0, 11, 28,
                                     36, 25, 32, 14, 22, 28, 20, 18, 4, 22, 35,
                                     19, 7, 8, 13, 5, 0, 8, 15, 15, 11, 4,
                                     39, 28, 26, 10, 27, 35, 18, 34, 30, 18, 23,
                                     1, 6, 30, 16, 26, 35, 9, 13, 6, 38, 39,
                                     8, 13, 7, 16, 22, 15, 25, 8, 35, 6, 17, 7,
                                     20, 25, 2, 16, 23, 24, 4, 36, 23, 30, 25])
        expected_counts = np.array([32, 36, 6, 21, 31, 13, 7, 24, 15, 18, 15,
                                    11, 38, 29, 1, 31, 24, 24, 3, 18, 3, 12, 38,
                                    35, 22, 5, 23, 32, 11, 20, 10, 37, 28, 2,
                                    27, 19, 25, 23, 20, 29, 3, 35, 39, 9, 9, 23,
                                    3, 26, 3, 31, 9, 10, 27, 7, 39, 21, 33, 34,
                                    34, 24, 33, 5, 36, 0, 11, 34, 5, 16, 8, 1,
                                    17, 35, 27, 36, 25, 3, 39, 35, 30, 29, 33,
                                    18, 17, 29, 20, 2, 5, 37, 12, 2, 27, 21, 39,
                                    39, 11, 22, 30, 17, 6, 7])
        aligned = al.pafft(counts, reference_counts, mzs)
        npt.assert_almost_equal(aligned, expected_counts)

    def test_aligns_signal_with_shifts(self):
        mzs = simulate_mzs(50)
        counts = 100 * np.sin(mzs)
        reference_counts = 100 * np.cos(mzs + 15)
        expected = np.array([
            59.7321531456132, 85.3529528081583, 99.9486175209072,
            -5.92657969171291, -5.92657969171291, -73.9674673665604,
            -13.8693197780068, 80.1921416688970, 48.2704538835641,
            -5.68663656246779, -99.6105062374414, -23.0601973134130,
            81.5227278930444, 69.8918487478728, -91.8829557682666,
            -99.4265499499549, -98.5879228687429, -98.5879228687429,
            -98.5879228687429, -98.5879228687429, -98.5879228687429,
            -11.4653555855311, -26.7984544429930, -96.5190803074683,
            76.4220585126419, 76.4220585126419, 76.4220585126419,
            -6.86003534271228, -96.7270291976866, 86.0303828152679,
            86.0303828152679, 86.0303828152679, 60.6519436858634,
            60.6519436858634, 60.6519436858634, 60.6519436858634,
            60.6519436858634, -87.0054923325986, 65.5607797531142,
            -38.2332065250098, -54.8049249401279, 50.5987779001395,
            84.1888141713609, -70.1386708071504, 96.4931946574145,
            -98.1518553180946, 95.9743204967746, 30.7178008411395,
            -78.7872186451279, 99.9647965615285])
        aligned = al.pafft(counts, reference_counts, mzs, 40., 40.)
        npt.assert_almost_equal(aligned, expected, 4)


if __name__ == '__main__':
    unittest.main()
