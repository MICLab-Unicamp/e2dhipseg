# -*- encoding: utf-8 -*-
# Module iaseshow

from numpy import *

def iaseshow(B, option="NORMAL"):
    from ia870 import iaisbinary, iaintersec, iagray, iabinary
    from ia870 import iasedil, iaero, iabshow

    option = option.upper()
    if option=='NON-FLAT':
        y = array([0],int32)
        if iaisbinary(B):
            B = iaintersec( iagray(B,'int32'),0)
    elif option=='NORMAL':
        if iaisbinary(B):    y=iabinary([1])
        else:
           y=array([0],int32)
    elif option=='EXPAND':
        assert iaisbinary(B), 'This option is only available with flat SE'
        y = iasedil( iabinary([1]),B)
        b1= iabinary(y>=0)
        b0= b1<0
        b0[shape(b0)[0]/2, shape(b0)[1]/2] = 1
        y = iabshow(b1,y,b0)
        return y
    else:
        print('iaseshow: not a valid flag: NORMAL, EXPAND or NON-FLAT')

    y = iasedil(y,B)
    return y

