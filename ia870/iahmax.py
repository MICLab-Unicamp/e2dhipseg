# -*- encoding: utf-8 -*-
# Module iahmax

from numpy import *
from ia870.iasecross import iasecross

def iahmax(f, h=1, Bc=iasecross()):
    from iasubm import iasubm
    from iainfrec import iainfrec

    g = iasubm(f,h)
    y = iainfrec(g,f,Bc);
    return y

