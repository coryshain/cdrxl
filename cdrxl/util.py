import sys


def stderr(s):
    sys.stderr.write('%s' % s)
    sys.stderr.flush()