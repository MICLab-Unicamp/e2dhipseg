# -*- encoding: utf-8 -*-
# Module iaasf

from numpy import *

def iaasf(f, SEQ="OC", b=None, n=1):
    from iasesum import iasesum
    from iaopen import iaopen
    from iaclose import iaclose
    from iasecross import iasecross
    if b is None:
        b = iasecross(None)

    SEQ = SEQ.upper()
    y = f
    if SEQ == 'OC':
        for i in range(1,n+1):
            nb = iasesum(b,i)
            y = iaopen( iaclose(y,nb),nb)
    elif SEQ == 'CO':
        for i in range(1,n+1):
            nb = iasesum(b,i)
            y = iaclose( iaopen(y,nb),nb)
    elif SEQ == 'OCO':
        for i in range(1,n+1):
            nb = iasesum(b,i)
            y = iaopen( iaclose( iaopen(y,nb),nb),nb)
    elif SEQ == 'COC':
        for i in range(1,n+1):
            nb = iasesum(b,i)
            y = iaclose( iaopen( iaclose(y,nb),nb),nb)

    return y

