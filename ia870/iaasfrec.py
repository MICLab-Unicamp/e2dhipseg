# -*- encoding: utf-8 -*-
# Module iaasfrec

from numpy import *
from ia870.iasecross import iasecross

def iaasfrec(f, SEQ="OC", b=iasecross(), bc=iasecross(), n=1):
    from iasesum import iasesum
    from iacloserec import iacloserec
    from iaopenrec import iaopenrec

    SEQ = SEQ.upper()
    y = f
    if SEQ == 'OC':
        for i in range(1,n+1):
            nb = iasesum(b,i)
            y = iacloserec(y,nb,bc)
            y = iaopenrec(y,nb,bc)
    elif SEQ == 'CO':
        for i in range(1,n+1):
            nb = iasesum(b,i)
            y = iaopenrec(y,nb,bc)
            y = iacloserec(y,nb,bc)
    else:
        assert 0,'Only accepts OC or CO for SEQ parameter'

    return y

