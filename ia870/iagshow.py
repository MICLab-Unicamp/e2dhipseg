# -*- encoding: utf-8 -*-
# Module iagshow

from numpy import *

def iagshow(X, X1=None, X2=None, X3=None, X4=None, X5=None, X6=None):
    from iaisbinary import iaisbinary
    from iagray import iagray
    from iaunion import iaunion
    from iaintersec import iaintersec
    from ianeg import ianeg
    from iaconcat import iaconcat

    if iaisbinary(X): X = iagray(X,'uint8')
    r = X
    g = X
    b = X
    if X1 is not None: # red 1 0 0
      assert iaisbinary(X1),'X1 must be binary overlay'
      x1 = iagray(X1,'uint8')
      r = iaunion(r,x1)
      g = iaintersec(g,ianeg(x1))
      b = iaintersec(b,ianeg(x1))
    if X2 is not None: # green 0 1 0
      assert iaisbinary(X2),'X2 must be binary overlay'
      x2 = iagray(X2,'uint8')
      r = iaintersec(r,ianeg(x2))
      g = iaunion(g,x2)
      b = iaintersec(b,ianeg(x2))
    if X3 is not None: # blue 0 0 1
      assert iaisbinary(X3),'X3 must be binary overlay'
      x3 = iagray(X3,'uint8')
      r = iaintersec(r,ianeg(x3))
      g = iaintersec(g,ianeg(x3))
      b = iaunion(b,x3)
    if X4 is not None: # magenta 1 0 1
      assert iaisbinary(X4),'X4 must be binary overlay'
      x4 = iagray(X4,'uint8')
      r = iaunion(r,x4)
      g = iaintersec(g,ianeg(x4))
      b = iaunion(b,x4)
    if X5 is not None: # yellow 1 1 0
      assert iaisbinary(X5),'X5 must be binary overlay'
      x5 = iagray(X5,'uint8')
      r = iaunion(r,x5)
      g = iaunion(g,x5)
      b = iaintersec(b,ianeg(x5))
    if X6 is not None: # cyan 0 1 1
      assert iaisbinary(X6),'X6 must be binary overlay'
      x6 = iagray(X6,'uint8')
      r = iaintersec(r,ianeg(x6))
      g = iaunion(g,x6)
      b = iaunion(b,x6)
    return iaconcat('d',r,g,b)
    return Y

