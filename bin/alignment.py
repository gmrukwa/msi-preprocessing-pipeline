from functools import partial
import gc
from multiprocessing import Pool
import sys

import numpy as np
from tqdm import tqdm

from components.spectrum.alignment import pafft


if __name__ == '__main__':
    MZS = sys.argv[1]
    REFERENCE = sys.argv[2]
    SPECTRA = sys.argv[3]
    POOL_SIZE = int(sys.argv[4])
    DESTINATION = sys.argv[5]
    mzs = np.loadtxt(MZS, delimiter=',')
    reference = np.loadtxt(REFERENCE, delimiter=',')
    spectra = np.load(SPECTRA, mmap_mode='r')
    align = partial(pafft, mzs=mzs, reference_counts=reference)
    with Pool(processes=POOL_SIZE) as pool:
        aligned = pool.map(
            align, tqdm(spectra, desc='Alignment'))
    del spectra
    gc.collect()
    with open(DESTINATION, 'wb') as out_file:
        np.save(out_file, aligned)
