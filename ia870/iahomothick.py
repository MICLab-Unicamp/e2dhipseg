# -*- encoding: utf-8 -*-
# Module iahomothick

from numpy import *

def iahomothick():
    from iase2hmt import iase2hmt
    from iabinary import iabinary


    Iab = iase2hmt( iabinary([[1,1,1],
                             [0,0,0],
                             [0,0,0]]),
                   iabinary([[0,0,0],
                             [0,1,0],
                             [1,1,1]]))


    return Iab

