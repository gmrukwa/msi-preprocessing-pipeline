"""Build GMM model over spectrum

There is recommendation for spectrum of 100 000 mass channels per 700-3500 Da.

Arguments:
    path to m/z axis
    path to spectrum
    path to result destination

"""
import pickle
import sys

import MatlabAlgorithms.MsiAlgorithms as msi
import matlab
import numpy as np


def as_matlab_type(array):
    return matlab.double([list(map(float, array.ravel()))])


if __name__ == '__main__':
    mzs = as_matlab_type(np.loadtxt(sys.argv[1]))
    spectrum = as_matlab_type(np.loadtxt(sys.argv[2]))
    engine = msi.initialize()
    print('Engine up!')
    model = engine.estimate_gmm(mzs, spectrum, nargout=1)
    mu = np.array(model['mu'], dtype=float).ravel()
    sig = np.array(model['sig'], dtype=float).ravel()
    w = np.array(model['w'], dtype=float).ravel()
    model = {
        'mu': mu,
        'sig': sig,
        'w': w,
        'KS': int(model['KS']),
        'meanspec': np.array(model['meanspec'], dtype=float).ravel()
    }
    engine.close()
    with open(sys.argv[3], mode='wb') as result_file:
        pickle.dump(model, result_file)
    np.savetxt('mu.txt', mu)
    np.savetxt('sig.txt', sig)
    np.savetxt('w.txt', w)
