# -*- encoding: utf-8 -*-
# Module iaopen

from numpy import *

def iaopen(f, b=None):
    from iadil import iadil
    from iaero import iaero
    from iasecross import iasecross
    if b is None:
        b = iasecross()

    y = iadil( iaero(f,b),b)

    return y

