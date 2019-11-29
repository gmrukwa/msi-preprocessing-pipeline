from functools import partial
import gc
from multiprocessing import Pool
import sys

import numpy as np
from tqdm import tqdm

from components.spectrum.baseline import adaptive_remove


if __name__ == '__main__':
    MZ_AXIS = sys.argv[1]
    SPECTRA = sys.argv[2]
    DESTINATION = sys.argv[3]
    POOL_SIZE = int(sys.argv[4])
    mz_axis = np.loadtxt(MZ_AXIS, delimiter=',')
    remover = partial(adaptive_remove, mz_axis)
    spectra = np.load(SPECTRA, mmap_mode='r')
    spectra = tqdm(spectra, desc='Baseline removal')
    with Pool(processes=POOL_SIZE) as pool:
        lowered = pool.map(remover, spectra, chunksize=800)
    del spectra
    gc.collect()
    with open(DESTINATION, 'wb') as outfile:
        np.save(outfile, lowered)
