# -*- encoding: utf-8 -*-
# Module iais

from numpy import *

def iais(f1, oper, f2=None, oper1=None, f3=None):
    from iaisbinary import iaisbinary
    from iaisequal import iaisequal
    from iaislesseq import iaislesseq
    from ianeg import ianeg
    from iathreshad import iathreshad
    from iabinary import iabinary


    if f2 == None:
        oper=oper.upper();
        if   oper == 'BINARY': return iaisbinary(f1)
        elif oper == 'GRAY'  : return not iaisbinary(f1)
        else:
            assert 0,'oper should be BINARY or GRAY, was'+oper
    elif oper == '==':    y = iaisequal(f1, f2)
    elif oper == '~=':    y = not iaisequal(f1,f2)
    elif oper == '<=':    y = iaislesseq(f1,f2)
    elif oper == '>=':    y = iaislesseq(f2,f1)
    elif oper == '>':     y = iaisequal( ianeg( iathreshad(f2,f1)),iabinary(1))
    elif oper == '<':     y = iaisequal( ianeg( iathreshad(f1,f2)),iabinary(1))
    else:
        assert 0,'oper must be one of: ==, ~=, >, >=, <, <=, it was:'+oper
    if oper1 != None:
        oper1 = oper1.upper()
        if   oper1 == '==': y = y and iaisequal(f2,f3)
        elif oper1 == '~=': y = y and (not iaisequal(f2,f3))
        elif oper1 == '<=': y = y and iaislesseq(f2,f3)
        elif oper1 == '>=': y = y and iaislesseq(f3,f2)
        elif oper1 == '>':  y = y and iaisequal( ianeg( iathreshad(f3,f2)),iabinary(1))
        elif oper1 == '<':  y = y and iaisequal( ianeg( iathreshad(f2,f3)),iabinary(1))
        else:
            assert 0,'oper1 must be one of: ==, ~=, >, >=, <, <=, it was:'+oper1


    return y

