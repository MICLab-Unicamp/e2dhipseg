# -*- encoding: utf-8 -*-
# Module iasereflect

from numpy import *

def iasereflect(Bi):
    from iaserot import iaserot


    Bo = iaserot(Bi, 180)


    return Bo

