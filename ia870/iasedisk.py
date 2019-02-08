# -*- encoding: utf-8 -*-
# Module iasedisk

from numpy import *

def iasedisk(r=3, DIM="2D", METRIC="EUCLIDEAN", FLAT="FLAT", h=0):
    from iabinary import iabinary
    from iasecross import iasecross
    from iasedil import iasedil
    from iasesum import iasesum
    from iasebox import iasebox
    from iaintersec import iaintersec
    from iagray import iagray

    METRIC = METRIC.upper()
    FLAT   = FLAT.upper()
    assert DIM=='2D','Supports only 2D structuring elements'
    if FLAT=='FLAT': y = iabinary([1])
    else:            y = int32([h])
    if r==0: return y
    if METRIC == 'CITY-BLOCK':
        if FLAT == 'FLAT':
            b = iasecross(1)
        else:
            b = int32([[-2147483647, 0,-2147483647],
                       [          0, 1,          0],
                       [-2147483647, 0,-2147483647]])
        return iasedil(y,iasesum(b,r))
    elif METRIC == 'CHESSBOARD':
        if FLAT == 'FLAT':
            b = iasebox(1)
        else:
            b = int32([[1,1,1],
                       [1,1,1],
                       [1,1,1]])
        return iasedil(y,iasesum(b,r))
    elif METRIC == 'OCTAGON':
        if FLAT == 'FLAT':
            b1,b2 = iasebox(1),iasecross(1)
        else:
            b1 = int32([[1,1,1],[1,1,1],[1,1,1]])
            b2 = int32([[-2147483647, 0,-2147483647],
                        [          0, 1,          0],
                        [-2147483647, 0,-2147483647]])
        if r==1: return b1
        else:    return iasedil( iasedil(y,iasesum(b1,r/2)) ,iasesum(b2,(r+1)/2))
    elif METRIC == 'EUCLIDEAN':
        v = arange(-r,r+1)
        x = resize(v, (len(v), len(v)))
        y = transpose(x)
        Be = iabinary(sqrt(x*x + y*y) <= (r+0.5))
        if FLAT=='FLAT':
            return Be
        be = h + int32( sqrt( maximum((r+0.5)*(r+0.5) - (x*x) - (y*y),0)))
        be = iaintersec( iagray(Be,'int32'),be)
        return be
    else:
        assert 0,'Non valid metric'


    return B

