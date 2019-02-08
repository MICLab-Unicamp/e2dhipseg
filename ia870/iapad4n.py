# -*- encoding: utf-8 -*-
# Module iapad4n

from numpy import *

def iapad4n(f, Bc, value, scale=1):
    from iaseshow import iaseshow

    if type(Bc) is not array:
      Bc = iaseshow(Bc)
    Bh, Bw = Bc.shape
    assert Bh%2 and Bw%2, 'structuring element must be odd sized'
    ch, cw = scale * Bh/2, scale * Bw/2
    g = value * ones( f.shape + scale * (array(Bc.shape) - 1))
    g[ ch: -ch, cw: -cw] = f
    y = g.astype( f.dtype.char)


    return y

