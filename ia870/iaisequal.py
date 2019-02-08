# -*- encoding: utf-8 -*-
# Module iaisequal

import numpy

def iaisequal(f1, f2, MSG=None):

    if f1.shape != f2.shape:
      return False
    return numpy.all(f1 == f2)

