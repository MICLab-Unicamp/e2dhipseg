# -*- encoding: utf-8 -*-
# Module iagdist

from numpy import *
from ia870.iasecross import iasecross

def iagdist(f, g, Bc=iasecross(), METRIC=None):
    from ianeg import ianeg
    from iagray import iagray
    from iaintersec import iaintersec
    from iaisequal import iaisequal
    from iacero import iacero
    from iaaddm import iaaddm
    from iaunion import iaunion

    assert METRIC is None,'Does not support EUCLIDEAN'
    fneg,gneg = ianeg(f),ianeg(g)
    y = iagray(gneg,'uint16',1)
    ero = iaintersec(y,0)
    aux = y
    i = 1
    while (ero != aux).any():
        aux = ero
        ero = iacero(gneg,fneg,Bc,i)
        y   = iaaddm(y,iagray(ero,'uint16',1))
        i   = i + 1
    y = iaunion(y,iagray(ero,'uint16'))

    return y

