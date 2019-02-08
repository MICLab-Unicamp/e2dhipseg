# -*- encoding: utf-8 -*-
# Module iaopentransf

from numpy import *
from ia870.iasecross import iasecross

def iaopentransf(f, type='OCTAGON', n=65535, Bc=iasecross(), Buser=iasecross()):
    from iaisbinary import iaisbinary
    from iabinary import iabinary
    from iaisequal import iaisequal
    from iaopen import iaopen
    from iasesum import iasesum
    from iasedisk import iasedisk
    from iaaddm import iaaddm
    from iagray import iagray
    from iagrain import iagrain
    from ialabel import ialabel

    assert iaisbinary(f),'Error: input image is not binary'
    type = type.upper()
    rec_flag = (type).find('-REC')
    if rec_flag != -1:
        type = type[:rec_flag] # remove the -rec suffix
    flag = not ((type == 'OCTAGON')  or
                (type == 'CHESSBOARD') or
                (type == 'CITY-BLOCK'))
    if not flag:
        n  = min(n,min(f.shape))
    elif  type == 'LINEAR-H':
        se = iabinary([1, 1, 1])
        n  = min(n,f.shape[1])
    elif  type =='LINEAR-V':
        se = iabinary([[1],[1],[1]])
        n  = min(n,f.shape[0])
    elif  type == 'LINEAR-45R':
        se = iabinary([[0, 0, 1],[0, 1, 0],[1, 0, 0]])
        n  = min(n,min(f.shape))
    elif  type == 'LINEAR-45L':
        se = iabinary([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
        n  = min(n,min(f.shape))
    elif  type == 'USER':
        se = Buser
        n  = min(n,min(f.shape))
    else:
        #print('Error: only accepts OCTAGON, CHESSBOARD, CITY-BLOCK, LINEAR-H, LINEAR-V, LINEAR-45R, LINEAR-45L, or USER as type, or with suffix -REC.')
        print('Error')
        return []
    k = 0
    y = uint16(zeros(f.shape))
    a = iabinary([1])
    z = iabinary([0])
    while not ( iaisequal(a,z) or (k>=n)):
        print('processing r=',k)
        if flag:
            a = iaopen(f,iasesum(se,k))
        else:
            a = iaopen(f,iasedisk(k,'2D',type))
        y = iaaddm(y, iagray(a,'uint16',1))
        k = k+1
    if rec_flag != -1:
        y = iagrain( ialabel(f,Bc),y,'max')

    return y

