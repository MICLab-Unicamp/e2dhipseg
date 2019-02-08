# -*- encoding: utf-8 -*-
# Module iase2interval

from numpy import *

def iase2interval(a, b):
    from ianeg import ianeg

    Iab = (a,ianeg(b))
    return Iab

