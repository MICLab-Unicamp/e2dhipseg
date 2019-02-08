# -*- encoding: utf-8 -*-
# Module iacenter

from numpy import *

def iacenter(f, b=None):
    from iaasf import iaasf
    from iaunion import iaunion
    from iaintersec import iaintersec
    from iaisequal import iaisequal
    from iasecross import iasecross
    if b is None:
        b = iasecross(None)

    y = f
    diff = 0
    while not diff:
        aux = y
        beta1 = iaasf(y,'COC',b,1)
        beta2 = iaasf(y,'OCO',b,1)
        y = iaunion( iaintersec(y,beta1),beta2)
        diff = iaisequal(aux,y)

    return y

