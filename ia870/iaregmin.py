# -*- encoding: utf-8 -*-
# Module iaregmin

from numpy import *
from ia870.iasecross import iasecross

def iaregmin(f, Bc=iasecross(), option="binary"):
    from iahmin import iahmin
    from iaaddm import iaaddm
    from iasubm import iasubm
    from iabinary import iabinary
    from iasuprec import iasuprec
    from iaunion import iaunion
    from iathreshad import iathreshad

    if option != "binary":
        raise Exception("iaregmin accepts only binary option")

    return iabinary(iasubm(iahmin(f,1,Bc), f))

