# -*- encoding: utf-8 -*-
# Module iaero

from numpy import *

def iaero(f, b=None):
    from ianeg import ianeg
    from iadil import iadil
    from iasereflect import iasereflect
    from iasecross import iasecross

    if b is None: b = iasecross()
    y = ianeg( iadil( ianeg(f),iasereflect(b)))
    return y

