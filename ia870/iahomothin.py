# -*- encoding: utf-8 -*-
# Module iahomothin

from numpy import *

def iahomothin():
    from iase2hmt import iase2hmt
    from iabinary import iabinary


    Iab = iase2hmt( iabinary([[0,0,0],
                             [0,1,0],
                             [1,1,1]]),
                   iabinary([[1,1,1],
                             [0,0,0],
                             [0,0,0]]))


    return Iab

