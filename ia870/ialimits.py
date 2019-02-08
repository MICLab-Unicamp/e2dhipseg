# -*- encoding: utf-8 -*-
# Module ialimits

from numpy import *

def ialimits(f):


    code = f.dtype
    if   code == bool:   y=array([0,1],'bool')
    elif code == uint8:  y=array([0,255],'uint8')
    elif code == uint16: y=array([0,(2**16)-1],'uint16')
    elif code == int32:  y=array([-((2**31)-1),(2**31)-1],'int32')
    elif code == int64:  y=array([-((2**63)-1), (2**63)-1],'int64')
    elif code == float64:y=array([-Inf,Inf],'float64')
    else:
        assert 0,'ialimits: Does not accept this typecode:%s' % code
    return y

