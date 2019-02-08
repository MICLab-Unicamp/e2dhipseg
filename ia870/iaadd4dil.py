# -*- encoding: utf-8 -*-
# Module iaadd4dil

from numpy import *

def iaadd4dil(f, c):
    from ia870 import ialimits

    if not c:
       return f
    if f.dtype == 'float64':
        y = f + c
    else:
        y = asarray(f,int64) + c
        k1,k2 = ialimits(f)
        y = ((f==k1) * k1) + ((f!=k1) * y)
        y = clip(y,k1,k2)
    a = y.astype(f.dtype)
    return a

