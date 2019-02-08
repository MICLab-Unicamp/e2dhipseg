# -*- encoding: utf-8 -*-
# Module iamat2set

from numpy import *

def iamat2set(A):
    from ia870 import ialimits


    if len(A.shape) == 1: A = A[newaxis,:]
    offsets = nonzero(ravel(A) ^ ialimits(A)[0])
    if type(offsets) == type(()):
        offsets = offsets[0]        # for compatibility with numarray
    if len(offsets) == 0: return ([],[])
    (h,w) = A.shape
    x = list(range(2))
    x[0] = offsets//w - (h-1)//2
    x[1] = offsets%w - (w-1)//2
    x = transpose(x)
    CV = x,ravel(A)[offsets]
    return CV

