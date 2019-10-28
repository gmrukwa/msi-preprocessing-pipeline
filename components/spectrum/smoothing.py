import numpy as np


def _generate_filter(window_size: int, order: int) -> np.ndarray:
    A = np.ones((window_size, order + 1))
    N = (window_size - 1) / 2
    window = np.arange(-N, N + 1)
    for k in range(order):
        A[:, k+1] = window ** (k+1)
    spectrum_filter, *_ = np.linalg.lstsq(np.matmul(A.T, A), A.T, rcond=None)
    return spectrum_filter[0]


def savitzky_golay(spectrum: np.ndarray, window: int=5, order: int=2) -> np.ndarray:
    spectrum_filter = _generate_filter(window, order)
    counts = np.convolve(spectrum, spectrum_filter, 'same')
    counts = np.clip(counts, a_min=0., a_max=None)
    size_difference = counts.size - spectrum.size
    assert size_difference >= 0
    offset = int(.5 + size_difference / 2)
    central_counts = counts[offset:offset+spectrum.size]
    assert central_counts.size == spectrum.size, (central_counts.size, spectrum.size)
    return central_counts
