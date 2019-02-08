# -*- encoding: utf-8 -*-
# Module iabshow

from numpy import *

def iabshow(f1, f2=None, f3=None, factor=17):
    from iabinary import iabinary
    from iaframe import iaframe
    from iadil import iadil
    from iaunion import iaunion
    from iasedisk import iasedisk
    from iaserot import iaserot
    from iasecross import iasecross
    from iasesum import iasesum

    assert f1.dtype == bool, 'f1 must be boolean image'
    factor = max(factor,9)
    hfactor = factor/2
    if size(f1.shape) == 1:
       f1 = f1[newaxis,:]
       if f2 != None: f2 = f2[newaxis,:]
       if f3 != None: f3 = f3[newaxis,:]
    bz = zeros(factor * array(f1.shape)).astype(bool)
    b0 = asarray(bz)
    b0[hfactor::factor,hfactor::factor] = f1
    fr1 = iaframe(zeros((factor,factor),bool))
    fr1 = iadil(b0,fr1)

    if f2 != None:
      assert f1.shape == f2.shape, 'f1 and f2 must have same shape'
      b1 = asarray(bz)
      b1[hfactor::factor,hfactor::factor] = f2
      fr2 = iadil(b1,iasedisk(hfactor - 4))
      fr1 = iaunion(fr1,fr2)

      if f3 != None:
        assert f1.shape == f3.shape, 'f1 and f3 must have same shape'
        bz[hfactor::factor,hfactor::factor] = f3
        fr3 = iadil(bz, iasesum(iaserot(iasecross(1),45),hfactor - 1))
        fr1 = iaunion(fr1,fr3)
    return fr1

