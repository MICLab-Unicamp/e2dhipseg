# -*- encoding: utf-8 -*-
# Module iaopenth

from numpy import *

def iaopenth(f, b=None):
    from iasubm import iasubm
    from iaopen import iaopen
    from iasecross import iasecross
    if b is None:
        b = iasecross()

    y = iasubm(f, iaopen(f,b))

    return y

