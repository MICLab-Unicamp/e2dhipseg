# -*- encoding: utf-8 -*-
# Module iaintershow

from numpy import *

def iaintershow(Iab):
    from iaseunion import iaseunion
    from iaintersec import iaintersec


    assert (type(Iab) is tuple) and (len(Iab) == 2),'not proper fortmat of hit-or-miss template'
    A,Bc = Iab
    S = iaseunion(A,Bc)
    Z = iaintersec(S,0)
    n = product(S.shape)
    one  = reshape(array(n*'1','c'),S.shape)
    zero = reshape(array(n*'0','c'),S.shape)
    x    = reshape(array(n*'.','c'),S.shape)
    saux = choose( S + iaseunion(Z,A), ( x, zero, one))
    s = ''
    for i in range(saux.shape[0]):
        s = s + ' \n'.join(list(saux[i]))

    return s

