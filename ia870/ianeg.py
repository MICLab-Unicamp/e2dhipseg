# -*- encoding: utf-8 -*-
# Module ianeg

from numpy import *

def ianeg(f):
    from ialimits import ialimits

    if ialimits(f)[0] == (- ialimits(f)[1]):
       y = -f
    else:
       y = ialimits(f)[0] + ialimits(f)[1] - f
    y = y.astype(f.dtype)
    return y

