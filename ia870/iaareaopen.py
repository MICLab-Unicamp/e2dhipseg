# -*- encoding: utf-8 -*-
# Module iaareaopen

import numpy as np
import ia870 as MT
from ia870.iasecross import iasecross
def iaareaopen_eq(f, a, Bc=iasecross()):

    if f.dtype == np.bool:
      fr = MT.ialabel(f,Bc)      # binary area open, use area measurement
      g = MT.iablob(fr,'area')
      y = g >= a
    else:
      y = np.zeros_like(f)
      k1 = f.min()
      k2 = f.max()
      for k in range(k1,k2+1):   # gray-scale, use thresholding decomposition
        fk = (f >= k)
        fo = MT.iaareaopen(fk,a,Bc)
        if not fo.any():
          break
        y = MT.iaunion(y, MT.iagray(fo,f.dtype,k))
    return y
def find_area(area, i):
    lista = []
    while area[i] >= 0:
        lista.append(i)
        i = area[i]
    area[lista] = i
    return i

def iaareaopen(f,a,Bc=iasecross()):
    a = -a
    s = f.shape
    g = np.zeros_like(f).ravel()
    f1 = np.concatenate((f.ravel(),np.array([0])))
    area = -np.ones((f1.size,), np.int32)
    N = MT.iaNlut(s, MT.iase2off(Bc))
    pontos = f1.nonzero()[0]
    pontos = pontos[np.lexsort((np.arange(0,-len(pontos),-1),f1[pontos]))[::-1]]
    for p in pontos:
        for v in N[p]:
            if f1[p] < f1[v] or (f1[p] == f1[v] and v < p):
                rv = find_area(area, v)
                if rv != p:
                    if area[rv] > a or f1[p] == f1[rv]:
                        area[p] = area[p] + area[rv]
                        area[rv] = p
                    else:
                        area[p] = a
    for p in pontos[::-1]:
        if area[p] >= 0:
            g[p] = g[area[p]]
        else:
            if area[p] <= a:
                g[p] = f1[p]
    return g.reshape(s)

