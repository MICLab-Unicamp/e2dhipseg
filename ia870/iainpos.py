# -*- encoding: utf-8 -*-
# Module iainpos

from numpy import *
from ia870.iasecross import iasecross

def iainpos(f, g, bc=iasecross()):
    from iaisbinary import iaisbinary
    from iagray import iagray
    from ianeg import ianeg
    from iadatatype import iadatatype
    from ialimits import ialimits
    from iasuprec import iasuprec
    from iaintersec import iaintersec
    from iaunion import iaunion

    assert iaisbinary(f),'First parameter must be binary image'
    fg = iagray( ianeg(f), iadatatype(g))
    k1 = ialimits(g)[1] - 1
    y = iasuprec(fg, iaintersec( iaunion(g, 1), k1, fg), bc)

    return y

