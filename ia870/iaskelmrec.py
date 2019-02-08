# -*- encoding: utf-8 -*-
# Module iaskelmrec

from numpy import *

def iaskelmrec(f, B=None):
    from iabinary import iabinary
    from iaintersec import iaintersec
    from iadil import iadil
    from iaunion import iaunion
    from iasecross import iasecross
    if B is None:
        B = iasecross(None)

    y = iabinary( iaintersec(f, 0))
    for r in range(max(ravel(f)),1,-1):
        y = iadil( iaunion(y,iabinary(f,r)), B)
    y = iaunion(y, iabinary(f,1))
    return y

