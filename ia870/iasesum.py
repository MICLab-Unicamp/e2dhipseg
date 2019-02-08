# -*- encoding: utf-8 -*-
# Module iasesum

import numpy as np

def iasesum(B, N=1):
    from ia870 import iaisbinary, iabinary, iasedil

    if N==0:
        if iaisbinary(B): return iabinary([1])
        else:             return array([0],int32) # identity
    NB = B
    for i in range(N-1):
        NB = iasedil(NB,B)
    return NB

