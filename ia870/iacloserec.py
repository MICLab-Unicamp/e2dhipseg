# -*- encoding: utf-8 -*-
# Module iacloserec

from numpy import *
from ia870.iasecross import iasecross

def iacloserec(f, bdil=iasecross(), bc=iasecross()):
    from iasuprec import iasuprec
    from iadil import iadil

    return iasuprec( iadil(f,bdil),f,bc)

