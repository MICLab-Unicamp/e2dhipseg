# -*- encoding: utf-8 -*-
# Module iathreshad

from numpy import *

def iathreshad(f, f1, f2=None):
    from iabinary import iabinary

    if f2 is None:
      y = (f1 <= f)
    else:
      y = ((f1 <= f) & (f <= f2))

    return y

