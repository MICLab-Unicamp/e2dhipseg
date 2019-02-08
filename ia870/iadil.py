# -*- encoding: utf-8 -*-
# Module iadil

from numpy import *

def iadil(f, b=None):
    from iamat2set import iamat2set
    from ialimits import ialimits
    from iaisbinary import iaisbinary
    from iaintersec import iaintersec
    from iagray import iagray
    from iaadd4dil import iaadd4dil
    from iasecross import iasecross

    if b is None: b = iasecross()

    if len(f.shape) == 1: f = f[newaxis,:]
    h,w = f.shape
    x,v = iamat2set(b)
    if len(x)==0:
        y = (ones((h,w)) * ialimits(f)[0]).astype(f.dtype)
    else:
        if iaisbinary(v):
            v = iaintersec( iagray(v,'int32'),0)
        mh,mw = max(abs(x)[:,0]),max(abs(x)[:,1])
        y = (ones((h+2*mh,w+2*mw)) * ialimits(f)[0]).astype(f.dtype)
        for i in range(x.shape[0]):
            if v[i] > -2147483647:
                y[mh+x[i,0]:mh+x[i,0]+h, mw+x[i,1]:mw+x[i,1]+w] = maximum(
                    y[mh+x[i,0]:mh+x[i,0]+h, mw+x[i,1]:mw+x[i,1]+w], iaadd4dil(f,v[i]))
        y = y[mh:mh+h, mw:mw+w]

    return y

