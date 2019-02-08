# -*- encoding: utf-8 -*-
# Module iaserot

from numpy import *

def iaserot(B, theta=45, DIRECTION="CLOCKWISE"):
    from iamat2set import iamat2set
    from iabinary import iabinary
    from iaset2mat import iaset2mat


    DIRECTION = DIRECTION.upper()
    if DIRECTION == "ANTI-CLOCKWISE":
       theta = -theta
    SA = iamat2set(B)
    theta = pi * theta/180
    (y,v)=SA
    if len(y)==0: return iabinary([0])
    x0 = y[:,1] * cos(theta) - y[:,0] * sin(theta)
    x1 = y[:,1] * sin(theta) + y[:,0] * cos(theta)
    x0 = int32((x0 +0.5)*(x0>=0) + (x0-0.5)*(x0<0))
    x1 = int32((x1 +0.5)*(x1>=0) + (x1-0.5)*(x1<0))
    x = transpose(array([transpose(x1),transpose(x0)]))
    BROT = iaset2mat((x,v))

    return BROT

