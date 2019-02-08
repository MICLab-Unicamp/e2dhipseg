# -*- encoding: utf-8 -*-
# Module iaglblshow

from numpy import *
from numpy.random import rand

def iaglblshow(X, border=0.0):
    from iastats import iastats
    from iaconcat import iaconcat

    iain = iastats(X,'min')
    iaax = iastats(X,'max')
    ncolors = iaax - iain + 1
    R = rand(ncolors)*255
    G = rand(ncolors)*255
    B = rand(ncolors)*255
    if iain == 0:
       R[0],G[0],B[0] = 0,0,0
    r=resize(take(R.astype(uint8), X.flat - iain),X.shape)
    g=resize(take(G.astype(uint8), X.flat - iain),X.shape)
    b=resize(take(B.astype(uint8), X.flat - iain),X.shape)
    Y=iaconcat('d',r,g,b)
    return Y

