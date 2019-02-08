# -*- encoding: utf-8 -*-
# Module iasecrop

def iasecrop(f):
    import numpy as np
    from ia870 import ialimits, iagray
    k1,k2 = ialimits(f)
    if not f.shape[0] & 1: f = np.vstack([k1 * np.ones((1,f.shape[1]),f.dtype),f])
    if not f.shape[1] & 1: f = np.hstack([k1 * np.ones((f.shape[0],1),f.dtype),f])
    h,w = f.shape
    h1 = (f != k1).max(1).nonzero()[0]
    dh = min([h1[0],h-h1[-1]-1])
    fh = f[dh:h-dh,:]
    w1 = (fh != k1).max(0).nonzero()[0]
    dw = min([w1[0],w-w1[-1]-1])
    return fh[:,dw:w-dw]

