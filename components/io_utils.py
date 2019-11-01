import os
from functools import partial

from functional import pipe
import numpy as np
import pandas as pd


def rooted_content(root: str):
    return [os.path.join(root, element) for element in os.listdir(root)]


subdirectories = pipe(rooted_content, partial(filter, os.path.isdir))
files = pipe(rooted_content, partial(filter, os.path.isfile))


def has_extension(path, extension: str='.txt'):
    return os.path.splitext(path)[1].lower() == extension


is_text = partial(has_extension, extension='.txt')
text_files = pipe(files, partial(filter, is_text))


def try_loadtxt(fname: str):
    try:
        return np.loadtxt(fname)
    except ValueError:
        return pd.read_csv(fname, delimiter=' ', header=None,
                           decimal=',').values
