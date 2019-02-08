# -*- encoding: utf-8 -*-
# Module iaedgeoff

from numpy import *
from ia870.iasecross import iasecross

def iaedgeoff(f, Bc=iasecross()):
    from iaframe import iaframe
    from iasubm import iasubm
    from iainfrec import iainfrec

    edge = iaframe(f)
    return iasubm( f, iainfrec(edge, f, Bc))

