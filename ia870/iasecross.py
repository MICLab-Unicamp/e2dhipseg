# -*- encoding: utf-8 -*-
# Module iasecross

from numpy import *

def iasecross(r=1):
    from ia870.iasesum import iasesum
    from ia870.iabinary import iabinary

    B = iasesum( iabinary([[0,1,0],
                           [1,1,1],
                           [0,1,0]]),r)
    return B

