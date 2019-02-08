# -*- encoding: utf-8 -*-
# Module iapad

from numpy import *
from ia870.iasecross import iasecross

def iapad(f, B=iasecross(), value=0):
    from iamat2set import iamat2set
    from iaseshow import iaseshow

    i,v=iamat2set( iaseshow(B));
    mni=i.min(axis=0)
    mxi=i.max(axis=0)
    f = asarray(f)
    if size(f.shape) == 1: f = f[newaxis,:]
    g = (value * ones(array(f.shape)+mxi-mni)).astype(f.dtype)
    g[-mni[0]:g.shape[0]-mxi[0], -mni[1]:g.shape[1]-mxi[1]] = f

    return g

