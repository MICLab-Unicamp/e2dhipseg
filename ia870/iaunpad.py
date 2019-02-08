# -*- encoding: utf-8 -*-
# Module iaunpad

from numpy import *
from ia870.iasecross import iasecross

def iaunpad(f, B=iasecross()):
    from iamat2set import iamat2set
    from iaseshow import iaseshow

    i,v=iamat2set( iaseshow(B));
    mni=minimum.reduce(i)
    mxi=maximum.reduce(i)
    g = f[-mni[0]:f.shape[0]-mxi[0], -mni[1]:f.shape[1]-mxi[1]]

    return g

