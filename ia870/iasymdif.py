# -*- encoding: utf-8 -*-
# Module iasymdif

from numpy import *

def iasymdif(f1, f2):
    from iaunion import iaunion
    from iasubm import iasubm
    y = iaunion( iasubm(f1,f2),iasubm(f2,f1))
    return y

