# -*- encoding: utf-8 -*-
# Module iamaxleveltype

from numpy import *

def iamaxleveltype(TYPE='uint8'):

    max = 0
    if   TYPE == 'uint8'  : max=255
    elif TYPE == 'binary' : max=1
    elif TYPE == 'uint16' : max=65535
    elif TYPE == 'int32'  : max=2147483647
    elif TYPE == 'float64'  : max=inf
    else:
        assert 0, 'does not support this data type:'+TYPE

    return max

