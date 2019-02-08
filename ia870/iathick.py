# -*- encoding: utf-8 -*-
# Module iathick

from numpy import *

def iathick(f, Iab=None, n=-1, theta=45, DIRECTION="CLOCKWISE"):
    from iaisbinary import iaisbinary
    from iaintersec import iaintersec
    from iasupgen import iasupgen
    from iainterot import iainterot
    from iaunion import iaunion
    from iaisequal import iaisequal
    from iahomothick import iahomothick
    if Iab is None:
        Iab = iahomothick()

    DIRECTION = DIRECTION.upper()
    assert iaisbinary(f),'f must be binary image'
    if n == -1: n = product(f.shape)
    y = f
    zero = iaintersec(f,0)
    for i in range(n):
        aux = zero
        for t in range(0,360,theta):
            sup = iasupgen( y, iainterot(Iab, t, DIRECTION))
            aux = iaunion( aux, sup)
            y = iaunion( y, sup)
        if iaisequal(aux,zero): break

    return y

