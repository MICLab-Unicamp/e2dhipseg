# -*- encoding: utf-8 -*-
# Module iasuprec

from numpy import *

def iasuprec(f, g, Bc=None):
    from iacero import iacero
    from iasecross import iasecross
    if Bc is None:
        Bc = iasecross(None)


    n = product(f.shape)
    y = iacero(f,g,Bc,n);

    return y

