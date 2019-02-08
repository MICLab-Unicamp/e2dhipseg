# -*- encoding: utf-8 -*-
# Module iaunion

from numpy import *

def iaunion(f1, f2, *args):

    y = maximum(f1,f2)
    for f in args:
        y = maximum(y,f)
    return y.astype(f1.dtype)

