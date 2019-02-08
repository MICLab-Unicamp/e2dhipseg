# -*- encoding: utf-8 -*-
# Module iaclose

from numpy import *

def iaclose(f, b=None):
    from iaero import iaero
    from iadil import iadil
    from iasecross import iasecross
    if b is None:
        b = iasecross()

    y = iaero( iadil(f,b),b)

    return y

