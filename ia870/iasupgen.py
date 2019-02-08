# -*- encoding: utf-8 -*-
# Module iasupgen

from numpy import *

def iasupgen(f, INTER):
    from iaintersec import iaintersec
    from iaero import iaero
    from ianeg import ianeg


    A,Bc = INTER
    y = iaintersec( iaero( f, A),
                   iaero( ianeg(f), Bc))

    return y

