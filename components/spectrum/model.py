from typing import Generator, NamedTuple, Tuple

import numpy as np
from scipy.stats import norm


class Matches:
    def __init__(self, indices, lengths):
        indices, lengths = np.ravel(indices), np.ravel(lengths)
        if indices.size != lengths.size:
            raise ValueError('indices and lengths should be equal size')
        self.indices, self.lengths = indices, lengths

    def __iter__(self):
        return zip(self.indices, self.lengths)

    def __len__(self):
        return self.indices.size


class Components:
    def __init__(self, means, sigmas, weights):
        means, sigmas, weights = np.ravel(means), np.ravel(sigmas), np.ravel(weights)
        if means.size != sigmas.size or means.size != weights.size:
            raise ValueError('means, sigmas and weights should be equal size')
        self.means, self.sigmas, self.weights = means, sigmas, weights

    def __getitem__(self, item):
        return Components(self.means[item], self.sigmas[item], self.weights[item])

    def __iter__(self):
        return zip(self.means, self.sigmas, self.weights)

    def __len__(self):
        return self.means.size


ComponentsGroups = NamedTuple('ComponentsGroups', [
    ('matches', Matches),
    ('new_components', Components)
])


def _match_size(components: Components, n_sigmas: int=4) -> int:
    limit = components.means[0] + n_sigmas * components.sigmas[0]
    return int(np.sum(components.means <= limit))


def _weight(components: Components) -> float:
    return float(np.sum(components.weights))


def _weighted_mean(components: Components, weight: float) -> float:
    return float(np.sum(components.weights * components.means) / weight)


def _sigma(components: Components, mean: float, weight: float) -> float:
    squares = components.means ** 2 + components.sigmas ** 2
    weighted_squares = components.weights * squares
    return np.sqrt(np.sum(weighted_squares) / weight - mean ** 2)


def _component_peak(mean: float, sigma: float, weight: float) -> float:
    return weight * norm.pdf(mean, loc=mean, scale=sigma)


def _highest_component_mean(components: Components) -> float:
    peaks = [_component_peak(*component) for component in components]
    highest = np.nonzero(np.max(peaks) == np.array(peaks))[0][0]
    return float(components.means[highest])


def _merged_component(chunk: Components) -> Tuple[float, float, float]:
    new_weight = _weight(chunk)
    temporary_mean = _weighted_mean(chunk, new_weight)
    new_sigma = _sigma(chunk, temporary_mean, new_weight)
    new_mean = _highest_component_mean(chunk)
    return new_mean, new_sigma, new_weight


def _make_chunks(components: Components, max_components: int, n_sigmas: int) \
        -> Generator[Tuple[int, int, float, float, float], None, None]:
    start = 0
    while start < len(components):
        temporary_end = min(start + max_components, len(components))
        size = _match_size(components[start:temporary_end], n_sigmas)
        chunk = components[start:start + size]
        mean, sigma, weight = _merged_component(chunk)
        yield start, size, mean, sigma, weight
        start += size


def merge(components: Components, max_components: int=4, n_sigmas: int=4) -> \
        ComponentsGroups:
    chunk_details = _make_chunks(components, max_components, n_sigmas)
    starts, sizes, means, sigmas, weights = zip(*chunk_details)
    matches = Matches(starts, sizes)
    components = Components(means, sigmas, weights)
    return ComponentsGroups(matches, components)


def apply_merging(data: np.ndarray, matches: Matches) -> np.ndarray:
    return np.hstack([
        np.sum(data[:, start: start+size], axis=1, keepdims=True)
        for start, size in matches
    ]).astype(dtype=np.float32)
