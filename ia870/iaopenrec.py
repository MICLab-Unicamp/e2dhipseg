# -*- encoding: utf-8 -*-
# Module iaopenrec

from numpy import *
from ia870.iasecross import iasecross

def iaopenrec(f, bero=iasecross(), bc=iasecross()):
    from iainfrec import iainfrec
    from iaero import iaero

    return iainfrec( iaero(f,bero),f,bc)

