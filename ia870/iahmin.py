# -*- encoding: utf-8 -*-
# Module iahmin

from numpy import *
from ia870.iasecross import iasecross

def iahmin(f, h=1, Bc=iasecross()):
    from iaaddm import iaaddm
    from iasuprec import iasuprec

    g = iaaddm(f,h)
    y = iasuprec(g,f,Bc);
    return y

