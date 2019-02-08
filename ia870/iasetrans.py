# -*- encoding: utf-8 -*-
# Module iasetrans

from numpy import *

def iasetrans(Bi, t):
    from ia870 import iamat2set
    from ia870 import iaset2mat


    x,v=iamat2set(Bi)
    Bo = iaset2mat((x+t,v))
    Bo = Bo.astype(Bi.dtype)

    return Bo

