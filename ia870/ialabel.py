# -*- encoding: utf-8 -*-
# Module ialabel

import numpy as np
import ia870 as MT
from ia870.iasecross import iasecross

def ialabel(f, Bc=iasecross()):
    return ialabel_unionfind(f,Bc)

# Implementation using morphological reconstruction iainfrec
def ialabel_rec(f, Bc=iasecross()):
    assert MT.iaisbinary(f),'Can only label binary image'
    faux=f.copy()
    label = 1
    y = MT.iagray( f,'uint16',0)          # zero image (output)
    x = faux.ravel().nonzero()            # get list of unlabeled pixel
    while len(x[0]):
        fmark = np.zeros_like(f)
        fmark.flat[x[0][0]] = 1           # get the first unlabeled pixel
        r = MT.iainfrec( fmark, faux, Bc) # detects all pixels connected to it
        faux -= r                         # remove them from faux
        r = MT.iagray( r,'uint16',label)  # label them with the value label
        y = MT.iaunion( y, r)             # merge them with the labeled image
        label = label + 1
        x = faux.ravel().nonzero()        # get list of unlabeled pixel
    return y

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

def ialabel_unionfind(img,Bc):
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
   O = np.zeros(imsize, dtype = int)
   cur_label = 0
   # pass 1: backward scan, forward neighbors
   for p in nonzero_nodes[::-1]:
      for nb in neighbors[p]:
         if img[nb]:
            Union(nb,p,parents)
   # pass 2_1: root labeled
   pbool = parents[nonzero_nodes] == nonzero_nodes
   labels = np.arange(1,pbool.sum()+1)
   O[nonzero_nodes[pbool]] = labels
   # pass 2_2: labeling root descendants
   for p in nonzero_nodes[~pbool]:
       O[p] = O[parents[p]]
   return O.reshape(imshape)

