# -*- encoding: utf-8 -*-
# Module iaskelm

from numpy import *
from ia870.iasecross import iasecross

def iaskelm(f, B=iasecross(), option="binary"):
    from iaisbinary import iaisbinary
    from ialimits import ialimits
    from iagray import iagray
    from iaintersec import iaintersec
    from iasesum import iasesum
    from iaero import iaero
    from iaisequal import iaisequal
    from iaopenth import iaopenth
    from iasedil import iasedil
    from iaunion import iaunion
    from iabinary import iabinary
    from iapad import iapad
    from iaunpad import iaunpad

    assert iaisbinary(f),'Input binary image only'
    option = option.upper()
    f = iapad(f,B)
    print(f)
    k1,k2 = ialimits(f)
    y = iagray( iaintersec(f, k1),'uint16')
    iszero = asarray(y)
    nb = iasesum(B,0)
    for r in range(1,65535):
        ero = iaero( f, nb)
        if iaisequal(ero, iszero): break
        f1 = iaopenth( ero, B)
        nb = iasedil(nb, B)
        y = iaunion(y, iagray(f1,'uint16',r))
    if option == 'BINARY':
        y = iabinary(y)
    y = iaunpad(y,B)
    return y

