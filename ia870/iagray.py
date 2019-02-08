# -*- encoding: utf-8 -*-
# Module iagray

from numpy import *

def iagray(f, TYPE="uint8", k1=None):
    from ia870 import iabinary, iais, iamaxleveltype, ianeg, ialimits

    #f = (f > 0)
    ff = array([0],TYPE)
    kk1,kk2 = ialimits(ff)
    if k1!=None:
        kk2=k1
    if   TYPE == 'uint8'  : y = where(f,kk2,kk1).astype(uint8)
    elif TYPE == 'uint16' : y = where(f,kk2,kk1).astype(uint16)
    elif TYPE == 'int32'  : y = where(f,kk2,kk1).astype(int32)
    elif TYPE == 'int64'  : y = where(f,kk2,kk1).astype(int64)
    elif TYPE == 'float64': y = where(f,kk2,kk1).astype(float64)
    else:
        assert 0, 'type not supported:'+TYPE
    return y

