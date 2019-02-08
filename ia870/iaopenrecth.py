# -*- encoding: utf-8 -*-
# Module iaopenrecth

from numpy import *

def iaopenrecth(f, bero=None, bc=None):
    from iasubm import iasubm
    from iaopenrec import iaopenrec
    from iasecross import iasecross
    if bero is None:
        bero = iasecross()
    if bc is None:
        bc = iasecross()

    y = iasubm(f, iaopenrec( f, bero, bc))

    return y

