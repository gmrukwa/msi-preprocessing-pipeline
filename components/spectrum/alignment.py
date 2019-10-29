from itertools import product


import numpy as np


def _segment_max_size(segment: np.ndarray):
    return int(segment.size / 20. + .5)


def _find_segment_cut_point(segment, reference_segment):
    """"""
    max_size = _segment_max_size(segment)
    segment_order = np.argsort(segment)
    reference_order = np.argsort(reference_segment)
    index_pairs = product(segment_order[:max_size], reference_order[:max_size])
    for segment_index, reference_index in index_pairs:
        if segment_index == reference_index:
            return segment_index + 1
    return segment_order[0] + 1


def _shift_segment(segment, shift):
    """"""
    if shift == 0 or np.abs(shift) >= segment.size:
        return segment
    if shift > 0:
        fillup = np.ones((shift,)) * segment[0]
        return np.hstack((fillup, segment[:segment.size - shift]))
    elif shift < 0:
        shift = -shift
        fillup = np.ones((shift,)) * segment[-1]
        return np.hstack((segment[shift:], fillup))


_BITS_LIMIT = 20
_MAGIC_PADDING = 1000000


def _estimate_padding(segment_size):
    bits = np.log2(segment_size)
    bits = int(bits+1)
    return 2 ** bits if bits <= _BITS_LIMIT else _MAGIC_PADDING + segment_size


def _convolve_with_reversed_time(segment, reference, padded_size):
    """Convolve segment and reference, flipping segment time axis"""
    transformed_reference = np.fft.fft(reference, n=padded_size)
    transformed_segment = np.fft.fft(segment, n=padded_size)
    transformed_convolution = \
        transformed_reference * np.conj(transformed_segment) / padded_size
    convolution = np.fft.ifft(transformed_convolution)
    convolution = np.real(convolution)
    return convolution


def _get_peak_within_shift_limit(convolution, shift_limit):
    if shift_limit < 1:
        return 0
    positive_shift_case = convolution[:shift_limit]
    negative_shift_case = convolution[-shift_limit:]
    highest_peaks = np.max(positive_shift_case), np.max(negative_shift_case)
    if max(highest_peaks) < .1:
        return 0
    if highest_peaks[0] > highest_peaks[1]:
        return np.argmax(positive_shift_case)
    else:
        return - negative_shift_case.size + np.argmax(negative_shift_case)


def _find_shift(segment, reference, shift_limit):
    """"""
    padded_size = _estimate_padding(segment.size)
    convolution = _convolve_with_reversed_time(segment, reference, padded_size)
    shift = _get_peak_within_shift_limit(convolution,
                                         min(shift_limit, padded_size))
    return shift


def _signals_chunks(counts, reference_counts, start, segment_size):
    end = start + max(segment_size * 2, 1)
    if end >= counts.size:
        segment = counts[start:]
        reference = reference_counts[start:]
    else:
        segment = counts[start + segment_size:end]
        reference = reference_counts[start + segment_size:end]
        cut_point = _find_segment_cut_point(segment, reference)
        end = start + cut_point + segment_size + 1
        segment = counts[start:end]
        reference = reference_counts[start:end]
    return segment, reference


def _shift_chunk_to_reference(segment, reference, mzs_increment, segment_mzs,
                              shift_limit):
    shift_scale = shift_limit / mzs_increment
    # new corner case
    mzs_index = min(int(.5 + segment.size / 2.), segment_mzs.size - 1)
    mz_shift = int(.5 + shift_scale * segment_mzs[mzs_index])
    shift = _find_shift(segment, reference, mz_shift)
    shifted = _shift_segment(segment, shift)
    return shifted


def pafft(counts, reference_counts, mzs, minimum_segment=.7, shift_limit=.1):
    """Alignment of spectral data using FFT correlation theorem

    Arguments:
        counts (ArrayLike) - counts to be aligned
        reference_counts (ArrayLike) - reference spectrum for alignment
        mzs (ArrayLike) - common m/z axis
        minimum_segment (float) - minimum segment size as percent of m/z
        shift_limit (float) - maximum shift for segment as percent of m/z

    Returns:
        Spectrum aligned to reference counts.

    Credits:
        Based on original algorithm by: Jason W. H. Wong
        Algorithm modified by: Michal Marczyk
        Implemented & fixed corner cases by: Grzegorz Mrukwa

    """
    assert counts.size == reference_counts.size
    assert counts.size == mzs.size
    minimum_segment *= .01
    shift_limit *= .01
    start = 0
    chunks = []
    while start < counts.size:
        if start == counts.size - 1:  # new corner case
            mzs_increment = mzs[start] - mzs[start - 1]
        else:
            mzs_increment = mzs[start + 1] - mzs[start]
        segment_scale = minimum_segment / mzs_increment
        segment_size = int(.5 + segment_scale * mzs[start])
        segment, reference = _signals_chunks(counts, reference_counts,
                                             start, segment_size)
        shifted = _shift_chunk_to_reference(segment, reference, mzs_increment,
                                            mzs[start:], shift_limit)
        chunks.append(shifted.astype(np.float32))
        assert segment.size == shifted.size, (segment.size, shifted.size)
        start += segment.size
    aligned = np.hstack(chunks)
    assert aligned.size == counts.size, (aligned.size, counts.size)
    return aligned
