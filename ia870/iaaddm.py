# -*- encoding: utf-8 -*-
# Module iaaddm

from numpy import *

def iaaddm(f1, f2):
    from ialimits import ialimits

    if type(f2) is array:
        assert f1.dtype == f2.dtype, 'Cannot have different datatypes:'
    k1,k2 = ialimits(f1)
    y = clip(f1.astype(int64)+f2, k1, k2)
    y = y.astype(f1.dtype)
    return y

