# -*- encoding: utf-8 -*-
# Module iathin

from numpy import *

def iathin(f, Iab=None, n=-1, theta=45, DIRECTION="CLOCKWISE"):
    import ia870 as MT

    if Iab is None: Iab = MT.iahomothin()

    DIRECTION = DIRECTION.upper()
    assert MT.iaisbinary(f),'f must be binary image'
    if n == -1: n = product(f.shape)
    y = f
    zero = MT.iaintersec(f,0)
    for i in range(n):
        aux = zero
        for t in range(0,360,theta):
            sup = MT.iasupgen( y, MT.iainterot(Iab, t, DIRECTION))
            aux = MT.iaunion( aux, sup)
            #y -= sup
            y = MT.iasubm( y, sup)
        #if iaisequal(aux,zero): break
        if not aux.any(): break
    return y

