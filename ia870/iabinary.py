# -*- encoding: utf-8 -*-
# Module iabinary

from numpy import *

def iabinary(f, k1=1):

    f = asarray(f)
    y = f >= k1
    return y

