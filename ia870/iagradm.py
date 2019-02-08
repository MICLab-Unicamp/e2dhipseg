# -*- encoding: utf-8 -*-
# Module iagradm

from numpy import *

def iagradm(f, Bdil=None, Bero=None):
    from iasubm import iasubm
    from iadil import iadil
    from iaero import iaero
    from iasecross import iasecross
    if Bdil is None: Bdil = iasecross()
    if Bero is None: Bero = iasecross()

    y = iasubm( iadil(f,Bdil),iaero(f,Bero))
    return y

