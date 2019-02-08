# -*- encoding: utf-8 -*-
# Module iaintersec

from numpy import *

def iaintersec(f1, f2, *args):

    y = minimum(f1,f2)
    for f in args:
        y = minimum(y,f)
    return y.astype(f1.dtype)

