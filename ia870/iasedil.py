# -*- encoding: utf-8 -*-
# Module iasedil

from numpy import *

def iasedil(B1, B2):
    from ia870 import ialimits, iagray, iabinary, iaisbinary, iamat2set, iaadd4dil, iasetrans, iaseunion

    assert (iaisbinary(B1) or (B1.dtype == int32) or (B1.dtype == int64) or (B1.dtype == float64)) and \
           (iaisbinary(B2) or (B2.dtype == int32) or (B2.dtype == int64) or (B2.dtype == float64)), \
           'iasedil: s.e. must be binary, int32, int64 or float64'
    if len(B1.shape) == 1: B1 = B1[newaxis,:]
    if len(B2.shape) == 1: B2 = B2[newaxis,:]
    if B1.dtype=='bool' and B2.dtype == 'bool':
       Bo = iabinary([0])
    else:
       Bo = array(ialimits(B1)[0]).reshape(1)
       if iaisbinary(B1):
          Bo = array(ialimits(B2)[0]).reshape(1)
          B1 = iagray(B1,B2.dtype,0)
       if iaisbinary(B2):
          Bo = array(ialimits(B1)[0]).reshape(1)
          B2 = iagray(B2,B1.dtype,0)
    x,v = iamat2set(B2)
    if len(x):
        for i in range(x.shape[0]):
            s = iaadd4dil(B1,v[i])
            st= iasetrans(s,x[i])
            Bo = iaseunion(Bo,st)
    return Bo

