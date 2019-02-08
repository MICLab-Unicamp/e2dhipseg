# -*- encoding: utf-8 -*-
# Module iaseintersec

from numpy import *

def iaseintersec(B1, B2):
    from ialimits import ialimits
    from ia870 import iamat2set, iaset2mat

    assert B1.dtype == B2.dtype, \
      'iaseintersec: Cannot have different datatypes: \
      %s and %s' % (str(B1.dtype), str(B2.dtype))
    type1 = B1.dtype
    #if len(B1) == 0: return B2
    if len(B1.shape) == 1: B1 = B1[newaxis,:]
    if len(B2.shape) == 1: B2 = B2[newaxis,:]
    if B1.shape != B2.shape:
        inf = ialimits(B1)[0]
        h1,w1 = B1.shape
        h2,w2 = B2.shape
        H,W = max(h1,h2),max(w1,w2)
        Hc,Wc = (H-1)/2,(W-1)/2    # center
        BB1,BB2 = asarray(B1),asarray(B2)
        B1, B2  = inf * ones((H,W)), inf *ones((H,W))
        dh1s , dh1e = (h1-1)/2 , (h1-1)/2 + (h1+1)%2 # deal with even and odd dimensions
        dw1s , dw1e = (w1-1)/2 , (w1-1)/2 + (w1+1)%2
        dh2s , dh2e = (h2-1)/2 , (h2-1)/2 + (h2+1)%2
        dw2s , dw2e = (w2-1)/2 , (w2-1)/2 + (w2+1)%2
        B1[ Hc-dh1s : Hc+dh1e+1  ,  Wc-dw1s : Wc+dw1e+1 ] = BB1
        B2[ Hc-dh2s : Hc+dh2e+1  ,  Wc-dw2s : Wc+dw2e+1 ] = BB2
    B = minimum(B1,B2).astype(type1)
    i,v = iamat2set(B)
    B = iaset2mat((i,v))
    return B

