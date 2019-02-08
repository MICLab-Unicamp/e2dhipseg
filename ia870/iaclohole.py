# -*- encoding: utf-8 -*-
# Module iaclohole

from numpy import *
from ia870.iasecross import iasecross

def iaclohole(f, Bc=iasecross()):
    from iaframe import iaframe
    from ianeg import ianeg
    from iainfrec import iainfrec

    delta_f = iaframe(f)
    return ianeg( iainfrec( delta_f, ianeg(f), Bc))

