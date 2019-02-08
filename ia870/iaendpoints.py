# -*- encoding: utf-8 -*-
# Module iaendpoints

from numpy import *

def iaendpoints(OPTION="LOOP"):
    from iase2hmt import iase2hmt
    from iabinary import iabinary


    Iab = None
    OPTION = OPTION.upper()
    if OPTION == 'LOOP':
        Iab = iase2hmt( iabinary([[0,0,0],
                                  [0,1,0],
                                  [0,0,0]]),
                        iabinary([[0,0,0],
                                  [1,0,1],
                                  [1,1,1]]))
    elif OPTION == 'HOMOTOPIC':
        Iab = iase2hmt( iabinary([[0,1,0],
                                  [0,1,0],
                                  [0,0,0]]),
                        iabinary([[0,0,0],
                                  [1,0,1],
                                  [1,1,1]]))


    return Iab

