# -*- encoding: utf-8 -*-
# Module iaregmax

from numpy import *
from ia870.iasecross import iasecross

def iaregmax(f, Bc=iasecross()):
    from iasubm import iasubm
    from iahmax import iahmax
    from iabinary import iabinary
    from iaregmin import iaregmin
    from ianeg import ianeg

    y = iasubm(f, iahmax(f,1,Bc))
    return iabinary(y)
    #return iaregmin( ianeg(f),Bc)

