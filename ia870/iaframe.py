# -*- encoding: utf-8 -*-
# Module iaframe

from numpy import *

def iaframe(f, WT=1, HT=1, DT=0, k1=None, k2=None):
    from iaunion import iaunion
    from iaintersec import iaintersec
    from ialimits import ialimits

    if k1 is None: k1 = ialimits(f)[1]
    if k2 is None: k2 = ialimits(f)[0]
    assert len(f.shape)==2,'Supports 2D only'
    y = iaintersec(f,k2)
    y[:,0:WT] = k1
    y[:,-WT:] = k1
    y[0:HT,:] = k1
    y[-HT:,:] = k1
    return y

