"""Cut M/Z axis

Arguments:
    maximal allowed m/z
    file with m/z axis
    destination of cut m/z axis
    location of datasets

"""

from functools import partial
import os
import sys

from functional import pipe
import numpy as np
from tqdm import tqdm

from components.utils import subdirectories


if __name__ == '__main__':
    mzs_limit = float(sys.argv[1])
    mzs = np.loadtxt(sys.argv[2])
    channels_count = int(np.sum(mzs <= mzs_limit))
    np.savetxt(sys.argv[3], mzs[:channels_count])
    for dataset in tqdm(subdirectories(sys.argv[4])):
        spectra = np.load(os.path.join(dataset, 'resampled_spectra.npy'))
        np.save(os.path.join(dataset, 'cut.npy'), spectra[:, :channels_count])
