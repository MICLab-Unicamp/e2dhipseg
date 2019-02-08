# -*- encoding: utf-8 -*-
# Module iatoggle

from numpy import *

def iatoggle(f, f1, f2, OPTION="GRAY"):
    from iabinary import iabinary
    from iasubm import iasubm
    from iagray import iagray
    from iaunion import iaunion
    from iaintersec import iaintersec
    from ianeg import ianeg


    y=iabinary( iasubm(f,f1),iasubm(f2,f))
    if OPTION.upper() == 'GRAY':
        t=iagray(y)
        y=iaunion( iaintersec( ianeg(t),f1),iaintersec(t,f2))


    return y

