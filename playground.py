'''
Place for random experiments, should never be imported anywhere
'''
import os
import glob
import numpy as np
from sys import argv
import cv2 as cv
from utils import ESC, Timer
import torch

import unet as u
import train_results as tr
from transforms import ReturnPatch, RandomFlip, Intensity, Noisify
from dataset import FloatHippocampusDataset

# Transforms test
if 'e2d' in argv:
    e2d = True
else:
    e2d = False
fd = FloatHippocampusDataset(orientation="axial", hiponly=True, e2d=e2d)

for i in range(len(fd)):
    image, mask = fd[i]

    
    
    rp = ReturnPatch(patch_size=(32, 32))
    flip = RandomFlip(modes=['horflip'])
    I = Intensity()
    g = Noisify()

    before = np.hstack([image[1], mask])
    cv.imshow("before", before)
    
    timer = Timer("Return Patch")
    timer.start()
    impatch, maskpatch = rp(image, mask, debug=False)
    impatch, maskpatch = flip(impatch, maskpatch)
    impatch, maskpatch = I(impatch, maskpatch)
    impatch, maskpatch = g(impatch, maskpatch)
    timer.stop()

    if e2d:
        for i in range(3):
            cv.imshow(str(i), impatch[i])
    
    after = np.hstack([impatch[1], maskpatch])
    cv.imshow("after", after)
    
    if cv.waitKey(0) == ESC:
        quit()
