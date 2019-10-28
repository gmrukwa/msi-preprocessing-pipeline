"""Merge components

Arguments:
    path to GMM model root
    path to merged model destination root

"""
from functools import partial
import os
import sys

from functional import pipe
import numpy as np

import components.spectrum.model as mdl
from components.utils import subdirectories


def load_components(components_root) -> mdl.Components:
    fname = partial(os.path.join, components_root)
    preserved = np.loadtxt(fname('indices_after_both.txt'), dtype=int)
    mu = np.loadtxt(fname('mu.txt'))[preserved].ravel()
    sig = np.loadtxt(fname('sig.txt'))[preserved].ravel()
    w = np.loadtxt(fname('w.txt'))[preserved].ravel()
    return mdl.Components(mu, sig, w)


def save_merged_model(destination_root, groups: mdl.ComponentsGroups):
    fname = partial(os.path.join, destination_root)
    np.savetxt(fname('merged_start_indices.txt'), groups.matches.indices, fmt="%i")
    np.savetxt(fname('merged_lengths.txt'), groups.matches.lengths, fmt="%i")
    np.savetxt(fname('merged_means.txt'), groups.new_components.means)
    np.savetxt(fname('merged_sigmas.txt'), groups.new_components.sigmas)
    np.savetxt(fname('merged_weights.txt'), groups.new_components.weights)


if __name__ == '__main__':
    components = load_components(sys.argv[1])
    merged = mdl.merge(components)
    print('New components: {0}'.format(merged.matches.indices.size))
    save_merged_model(sys.argv[2], merged)
