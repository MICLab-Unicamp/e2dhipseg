# -*- encoding: utf-8 -*-
# Module iaconcat

from numpy import *

def iaconcat(DIM, X1, X2, X3=None, X4=None):

    aux = 'newaxis,'
    d = len(X1.shape)
    if d < 3: X1 = eval('X1[' + (3-d)*aux + ':]')
    d1,h1,w1 = X1.shape
    d = len(X2.shape)
    if d < 3: X2 = eval('X2[' + (3-d)*aux + ':]')
    d2,h2,w2 = X2.shape
    h3 = w3 = d3 = h4 = w4 = d4 = 0
    if X3 != None:
       d = len(X3.shape)
       if d < 3: X3 = eval('X3[' + (3-d)*aux + ':]')
       d3,h3,w3 = X3.shape
    if X4 != None:
       d = len(X4.shape)
       if d < 3: X4 = eval('X4[' + (3-d)*aux + ':]')
       d4,h4,w4 = X4.shape
    h = [h1, h2, h3, h4]
    w = [w1, w2, w3, w4]
    d = [d1, d2, d3, d4]
    if DIM in ['WIDTH', 'W', 'w', 'width']:
       hy, wy, dy = max(h), sum(w), max(d)
       Y = zeros((dy,hy,wy),X1.dtype)
       Y[0:d1, 0:h1, 0 :w1   ] = X1
       Y[0:d2, 0:h2, w1:w1+w2] = X2
       if X3 != None:
          Y[0:d3, 0:h3, w1+w2:w1+w2+w3] = X3
          if X4 != None:
              Y[0:d4, 0:h4, w1+w2+w3::] = X4
    elif DIM in ['HEIGHT', 'H', 'h', 'height']:
       hy, wy, dy = sum(h), max(w), max(d)
       Y = zeros((dy,hy,wy),X1.dtype)
       Y[0:d1, 0 :h1   , 0:w1] = X1
       Y[0:d2, h1:h1+h2, 0:w2] = X2
       if X3 != None:
           Y[0:d3, h1+h2:h1+h2+h3, 0:w3] = X3
           if X4 != None:
               Y[0:d4, h1+h2+h3::, 0:w4] = X4
    elif DIM in ['DEPTH', 'D', 'd', 'depth']:
       hy, wy, dy = max(h), max(w), sum(d)
       Y = zeros((dy,hy,wy),X1.dtype)
       Y[0:d1    , 0:h1, 0:w1   ] = X1
       Y[d1:d1+d2, 0:h2, 0:w2] = X2
       if X3 != None:
           Y[d1+d2:d1+d2+d3, 0:h3, 0:w3] = X3
           if X4 != None:
               Y[d1+d2+d3::, 0:h4, 0:w4] = X4
    if Y.shape[0] == 1: # adjustment
       Y = Y[0,:,:]
    return Y

