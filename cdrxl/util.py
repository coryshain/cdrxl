import sys
import os
import pickle


def stderr(s):
    sys.stderr.write('%s' % s)
    sys.stderr.flush()


def load_cdrxl(path):
    with open(os.path.join(path, 'model', 'm.obj'), 'rb') as f:
        m = pickle.load(f)
    m.load(path)
    return m