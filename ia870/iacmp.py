# -*- encoding: utf-8 -*-
# Module iacmp

from numpy import *

def iacmp(f1, oper, f2, oper1=None, f3=None):
    from iaintersec import iaintersec
    from iabinary import iabinary


    if   oper == '==':    y = (f1==f2)
    elif oper == '~=':    y = (f1!=f2)
    elif oper == '<=':    y = (f1<=f2)
    elif oper == '>=':    y = (f1>=f2)
    elif oper == '>':     y = (f1> f2)
    elif oper == '<':     y = (f1< f2)
    else:
        assert 0, 'oper must be one of: ==, ~=, >, >=, <, <=, it was:'+oper
    if oper1 != None:
        if   oper1 == '==':     y = iaintersec(y, f2==f3)
        elif oper1 == '~=':     y = iaintersec(y, f2!=f3)
        elif oper1 == '<=':     y = iaintersec(y, f2<=f3)
        elif oper1 == '>=':     y = iaintersec(y, f2>=f3)
        elif oper1 == '>':      y = iaintersec(y, f2> f3)
        elif oper1 == '<':      y = iaintersec(y, f2< f3)
        else:
            assert 0, 'oper1 must be one of: ==, ~=, >, >=, <, <=, it was:'+oper1

    y = iabinary(y)


    return y

