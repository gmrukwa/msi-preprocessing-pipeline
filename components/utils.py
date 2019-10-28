import os
from functools import partial

from functional import pipe

only_directories = partial(filter, os.path.isdir)


def rooted_listdir(root):
    return [os.path.join(root, path) for path in os.listdir(root)]


subdirectories = pipe(rooted_listdir, only_directories, list)
