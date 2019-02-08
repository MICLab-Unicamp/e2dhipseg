# -*- encoding: utf-8 -*-
# Module iaseline

from numpy import *

def iaseline(l=3, theta=0):
    from iaset2mat import iaset2mat
    from iabinary import iabinary

    theta = pi*theta/180
    if abs(tan(theta)) <= 1:
        s  = sign(cos(theta))
        x0 = arange(0, l * cos(theta)-(s*0.5),s)
        x1 = floor(x0 * tan(theta) + 0.5)
    else:
        s  = sign(sin(theta))
        x1 = arange(0, l * sin(theta) - (s*0.5),s)
        x0 = floor(x1 / tan(theta) + 0.5)
    x = transpose(array([x1,x0],int32))
    B = iaset2mat((x,iabinary(ones((x.shape[1],1)))))
    return B

