'''
Defines custom losses and metrics, with tests when running directly (python3 metrics.py)

Author: Diedre Carmo
https://github.com/dscarmo
'''
from sys import argv
import sys
import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
from transforms import ToNumpy
from utils import random_rectangle, ESC

def vol_dice(inpt, target, smooth=1.0):
    '''
    Calculate DICE of volume
    '''
    q = inpt.size(0)
    assert len(inpt) != 0, " trying to compute DICE of nothing"

    iflat = inpt.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    eps = 0
    if smooth == 0.0:
        eps = sys.float_info.epsilon
    
    iflat_sum = iflat.sum()
    tflat_sum = tflat.sum()

    if iflat_sum.item() == 0.0 and tflat_sum.item() == 0.0:
        print("DICE Metric got black mask and prediction!")
        dice = torch.tensor(1.0, requires_grad=True, device=inpt.device)
    else:
        dice = (2. * intersection + smooth) / (iflat_sum + tflat_sum + smooth + eps)

    value = dice.item()
    assert value >= 0.0 or value <= 1.0, " DICE not between 0 and 1! something is wrong"
    
    return dice

def batch_dice(inpt, target, smooth=1.0):
    '''
    Calculate DICE of a batch of two binary masks
    Returns mean dice of all slices
    '''
    q = inpt.size(0)
    assert len(inpt) != 0, " trying to compute DICE of nothing"

    iflat = inpt.contiguous().view(q, -1)
    tflat = target.contiguous().view(q, -1)
    intersection = (iflat * tflat).sum(dim=1)
    
    eps = 0
    if smooth == 0.0:
        eps = sys.float_info.epsilon
    
    iflat_sum = iflat.sum(dim=1)
    tflat_sum = tflat.sum(dim=1)
    
    dice = (2. * intersection + smooth) / (iflat_sum + tflat_sum + smooth + eps)
    
    dice = dice.mean()
    value = dice.item()
    assert value >= 0.0 or value <= 1.0, " DICE not between 0 and 1! something is wrong"
    
    return dice

class DICEMetric():
    '''
    Calculates DICE Metric
    '''
    def __init__(self, apply_sigmoid=False, mask_ths=0.5):
        self.apply_sigmoid = apply_sigmoid
        self.mask_ths = mask_ths
        print("DICE Metric initialized with apply_sigmoid={}".format(apply_sigmoid))

    def __call__(self, probs, target):
        '''
        Returns only DICE metric, as volumetric dice
        probs: output of last convolution, sigmoided or not (use apply_sigmoid=True if not)
        targets: float binary target mask
        '''
        if self.apply_sigmoid:
            probs = probs.sigmoid()

        p_min = probs.min()
        p_max = probs.max()
        assert p_max <= 1.0 and p_min >= 0.0, "FATAL ERROR: DICE metric input not bounded! Did you apply sigmoid?"

        #mask = ((probs - p_min)/(p_max - p_min) > self.mask_ths).float() # were giving bad results
        mask = (probs > self.mask_ths).float()
        return vol_dice(mask, target, smooth=0.0).item()

class DICELoss(nn.Module):
    '''
    Calculates DICE Loss
    '''
    def __init__(self, weight=None, size_average=True, apply_sigmoid=False, volumetric=False):
        super(DICELoss, self).__init__()
        self.apply_sigmoid = apply_sigmoid
        self.volumetric = volumetric
        print("DICE Loss initialized with apply_sigmoid={}, volumetric={}".format(apply_sigmoid, self.volumetric))

    def forward(self, probs, targets):
        '''
        probs: output of last convolution, sigmoided or not (use apply_sigmoid=True if not)
        targets: binary target mask
        '''
        if self.apply_sigmoid:
            probs = probs.sigmoid()
        
        p_min = probs.min()
        p_max = probs.max()
        assert p_max <= 1.0 and p_min >= 0.0, "FATAL ERROR: DICE loss input not bounded! Did you apply sigmoid?"

        q = targets.size(0)
        if self.volumetric:
            score = vol_dice(probs, targets)
        else:
            score = batch_dice(probs, targets)
        loss = 1 - score
        return loss

def dice_test(batch_size=1, waitms=0, show=True):
    '''
    Randomly create masks and show dice, looking for inconsistencys
    '''
    print("Performing random rectangles test")
    print("Press ESC to interrupt test")
    key = None
    size = (128*3, 128*3) # width, height format
    dice_loss = DICELoss()
    dice_metric = DICEMetric()
    while key != ESC:
        try:
            # batch_size, num_channels, nrows, ncols
            batch_a = torch.zeros((batch_size, 1, size[1], size[0]))
            batch_b = torch.zeros((batch_size, 1, size[1], size[0]))
            for i in range(batch_size):    
                a = torch.zeros((1, size[1], size[0])) 
                b = torch.zeros((1, size[1], size[0]))
                
                x, y, w, h = random_rectangle(size)
                rect_a = (x, y, w, h)
                a[:, y:y+h, x:x+w] = 1

                x, y, w, h = random_rectangle(size)
                rect_b = (x, y, w, h)
                b[:, y:y+h, x:x+w] = 1
                batch_a[i] = a
                batch_b[i] = b
                if show:
                    cv.imshow("A mask", a.squeeze().numpy())
                    cv.imshow("B mask", b.squeeze().numpy())

            

            loss = dice_loss(batch_a, batch_b)
            metric = dice_metric(batch_a, batch_b)
            print("Batch size: {}, loss: {}, metric: {}".format(batch_size, loss, metric))
            
            if show:
                display = np.zeros((size))
            
                x, y, w, h = rect_a
                cv.rectangle(display, (x, y), (x+w, y+h), 0.5)
                
                x, y, w, h = rect_b
                cv.rectangle(display, (x, y), (x+w, y+h), 1)

                text_row = 9*size[1]//10
                cv.putText(display, "m:" + str(round(metric, 2)), (size[0]//10, text_row), cv.FONT_HERSHEY_SIMPLEX, 1, 1)
                cv.putText(display, "l:" + str(round(loss.item(), 2)), (5*size[0]//10, text_row), cv.FONT_HERSHEY_SIMPLEX, 1, 1)
                cv.imshow("DICE Test, m is metric and l is smooth loss", display)
                key = cv.waitKey(waitms)
        except KeyboardInterrupt:
            quit()
        
    print("DICE test interrupted by ESC")

def random_test():
    print("Performing random DICE test looking for inconsistencies... Press CTRL+C to continue to next test.")
    while True:
        try:
            a = torch.rand((30, 1, 128, 128))
            b = torch.rand((30, 1, 128, 128))
            q = a.size(0)
            dice = batch_dice(a, b)
            print(dice)
            assert dice >= 0 and dice <= 1.0
        except KeyboardInterrupt:
            return

def vol_test():
    '''
    Specific dice test for the volumetric case
    '''
    dice = DICEMetric(apply_sigmoid=False, mask_ths=0.5)
    from dataset import FloatHippocampusDataset
    from utils import viewnii
    from transforms import Compose, CenterCrop, ToTensor, ToNumpy
    import random
    db = FloatHippocampusDataset(mode="train", transform=Compose([CenterCrop(160, 160, 160), ToTensor()]), return_volume=True, verbose=False)
    _, mask = db[10]
    test_shape = torch.randn_like(mask)
    test_shape = (test_shape - test_shape.min()) / (test_shape.max() - test_shape.min())
    #viewnii(mask.squeeze().numpy(), test_shape.squeeze().numpy())
    print(dice(test_shape, mask))
    print(dice(mask, mask))
    print(dice(torch.ones_like(mask), mask))
    print(dice(torch.zeros_like(mask), mask))


if __name__ == "__main__":
    if len(argv) > 1:
        if argv[1] == "random":
            random_test()
        elif argv[1] == "rect":        
            dice_test(batch_size=1, show=False)
        elif argv[1] == "volumetric":
            vol_test()
        else:
            print("Unrecognized arguments {}".format(str(argv)))
    else:
        random_test()
        dice_test()
