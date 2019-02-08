# -*- encoding: utf-8 -*-
# Module ialabelflat

import numpy as np
import ia870 as MT
from ia870.iasecross import iasecross

def ialabelflat(f, Bc=iasecross(), delta=0):
    fr = f.astype(np.int) + 1
    return ialabelflat_unionfind(fr,Bc,delta)

# implementation by union find
def Find(x,parents): # uses path compression
   if parents[x] == x:
      return x
   else:
      parents[x] = Find(parents[x],parents)
      return parents[x]

def Union(n, p,parents):
   r = Find(n,parents)
   if r != p:
      parents[r] = p

def ialabelflat_unionfind(img,Bc,delta):
   imshape = img.shape
   imsize  = img.size

   # Offsets and neighbors
   offsets = MT.iase2off(Bc,'fw')
   neighbors = MT.iaNlut(imshape,offsets)

   parents = np.arange(imsize,dtype = int)

   # Raster image and no-nzero pixels
   img = img.ravel()
   nonzero_nodes = np.nonzero(img)[0]
   img =  np.concatenate((img,np.array([0])))
   g = np.zeros(imsize, dtype = int)
   cur_label = 0
   # pass 1: backward scan, forward neighbors
   for p in nonzero_nodes[::-1]:
      v = img[p]
      for nb in neighbors[p]:
         if img[nb] and (abs(v - img[nb]) <= delta):
            Union(nb,p,parents)
   # pass 2_1: root labeled
   pbool = parents[nonzero_nodes] == nonzero_nodes
   labels = np.arange(1,pbool.sum()+1)
   g[nonzero_nodes[pbool]] = labels
   # pass 2_2: labeling root descendants
   for p in nonzero_nodes[~pbool]:
       g[p] = g[parents[p]]
   return g.reshape(imshape)

