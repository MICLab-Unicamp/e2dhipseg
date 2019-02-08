# -*- encoding: utf-8 -*-
# Module iaareaclose

from numpy import *
from ia870.iasecross import iasecross

def iaareaclose(f, a, Bc=iasecross()):
    from ianeg import ianeg
    from iaareaopen import iaareaopen

    y = ianeg( iaareaopen( ianeg(f),a,Bc))
    return y

