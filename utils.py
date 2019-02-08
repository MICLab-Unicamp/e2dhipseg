'''
Utility functions and constants that dont belong to a specific module

Author: Diedre Carmo
https://github.com/dscarmo
'''
from random import randint
import os
import math
import numpy as np
import torch
from torch.cuda import get_device_name
import cv2 as cv
from matplotlib import pyplot as plt
import time
from skimage.transform import rotate

orientations = ["sagital", "coronal", "axial"]

ESC = 27
UP = 82
DOWN = 84
NUP = 56
NDOWN = 50
STOP = 115
REVERSE = 114
BIG = 98
VSMALL = 118

def tag2numpy(shape, file_tag):
    '''
    Interpret tag file segmentation
    '''
    points = np.vstack([file_tag[:,:3]])

    pts_calculados  = np.zeros(file_tag.shape).astype(int)
    pts_calculados[:,0] = points[:,0] + (shape[2] / 2)
    pts_calculados[:,1] = points[:,1] + (shape[0] / 2)
    pts_calculados[:,2] = points[:,2]+ (shape[1] / 2)

    esquerda_direita    = pts_calculados[:,0]
    anterior_posterior  = pts_calculados[:,1]
    inferior_superior   = pts_calculados[:,2]

    seg_mask = np.zeros(shape).astype('bool')
    seg_mask[anterior_posterior, inferior_superior, esquerda_direita] = True

    return seg_mask

def plots(ths_study, cons, study_ths, opt="", savepath=None, mean_result=0, name=""):
    '''
    Stuff Plotting
    '''
    if study_ths:
        print(ths_study)
        plt.figure(num="THS Study")
        plt.title("Each threshold for all test volumes")
        plt.xlabel("Threshold")
        plt.ylabel("DICE")
        plt.boxplot(ths_study, labels=np.array(range(1, 10))/10)
    if np.array(cons).sum() > 0:
        print(cons)
        with open(os.path.join(savepath, opt + str(mean_result) + "cons.txt"), 'a') as f:
            f.write(str(cons))
            f.write("Final result: " + str(mean_result))
        plt.figure(num=name)
        plt.title("Individual VS Consensus DICE")
        plt.ylabel("DICE")
        plt.ylim(-0.1)
        labels=["Sagital", "Coronal", "Axial", "Consensus"]
        plt.boxplot(cons, labels=labels)
        plt.tight_layout()
        if savepath is not None:
            print("Saving consensus")
            plt.savefig(os.path.join(savepath, opt + "cons.eps"), format="eps", dpi=1000)
        else:
            print("Warning: Consensus not saved")

def normalizeMri(data):
    '''
    Normalizes float content on MRI samples to 0-1, using min-max normalization
    
    data: input 3D numpy data do be normalized
    returns: normalized 3D numpy data 
    '''
    img = np.zeros((data.shape), dtype=data.dtype)
    cv.normalize(data, img, alpha=1.0, beta=0.0, norm_type=cv.NORM_MINMAX)
    return img

def viewnii(sample, mask=None, id="", wait=1, rotate=90, mrotate=None, quit_on_esc=True, fx=2, fy=2, customshape=None):
    '''
    Simple visualization of all orientations of mri
    '''
    if customshape is None:
        shape = sample.shape
    else:
        shape = customshape

    if mrotate is None:
        mrotate = rotate
    kup = True
    for j in range(0,3):
        #for k in range(0, shape[j]):
        k = 0
        while k >= 0 and k < shape[j]:
            if j == 0:
                view = myrotate(sample[k,:,:], rotate)
                if mask is not None:
                    mview = myrotate(mask[k,:,:], mrotate)
            elif j == 1:
                view = myrotate(sample[:,k,:], rotate)
                if mask is not None:
                    mview = myrotate(mask[:,k,:], mrotate)
            elif j == 2:
                view = myrotate(sample[:,:,k], rotate)
                if mask is not None:
                    mview = myrotate(mask[:,:,k], mrotate)

            view = cv.resize(view, (0,0), fx=fx, fy=fy)
            imagePrint(view, orientations[j] + " vol " + str(wait) + "ms", scale=0.8)
            if mask is not None: 
                mview = cv.resize(mview, (0,0), fx=fx, fy=fy)
                try:
                    sum_view = view + mview
                    sum_view[sum_view > 1] = 1
                except ValueError as ve:
                    print("Mask and vol dont match, skipping!!!")
                    break
                imagePrint(mview, orientations[j] + " mask ", scale=0.8)
                imagePrint(sum_view, orientations[j] + " overlap ", scale=0.8)

                final_view = np.hstack([view, mview, sum_view])
            else:
                final_view = view
            cv.imshow(id, final_view)

            key = cv.waitKey(wait)
            if key == ESC:
                if quit_on_esc:
                    quit()
                else:
                    return
            elif key == UP or key == NUP:
                if wait != 0:
                    inc = int(wait/10)
                    if inc == 0: inc = 1
                    wait += inc
                else:
                    wait = 1
            elif key == DOWN or key == NDOWN:
                if wait > 0:
                    dec = int(wait/10)
                    if dec == 0: dec = 1
                    wait -= dec
                if wait <= 0:
                    wait = 0
            elif key == STOP:
                wait = 0
            elif key == REVERSE:
                kup = not kup
            elif key == BIG:
                fx = fx + fx/2
                fy = fy + fy/2
            elif key == VSMALL:
                fx = fx - fx/2
                fy = fy - fy/2

            if kup:
                k += 1
            else:
                k -= 1
            

def myrotate(inpt, d):
    '''
    Simplifies rotation calls
    '''
    return rotate(inpt, d, resize=True)

def check_name(word, basename, char='-'):
    '''
    Check presence of an argument in arg string
    '''
    return word in basename.split(char)


def cprint(msg, out=True):
    '''
    Conditional print
    '''
    if out:
        print(msg)

class Timer():
    def __init__(self, name ='', ndecimals=10):
        self.ndecimals = ndecimals
        self.name = name
    
    def start(self):
        '''
        Starts counting
        '''
        self.begin = time.time()

    def stop(self, printout=True):
        '''
        Tries to print and returns float time passed 
        '''
        end = round(time.time() - self.begin, self.ndecimals)
        if printout: print("Timer {}: {}s".format(self.name, str(end))) 
        return end

def show_grayimage(title, npimage, show=False):
    '''
    Plt gray img display
    '''
    plt.figure(num=title)
    plt.imshow(npimage, cmap='gray')
    plt.title(title)
    if show:
        plt.show()

def imagePrint(img, st, org=(20, 20), scale=1, color=1, font=cv.FONT_HERSHEY_SIMPLEX):
    '''
    Simplifies printing text on images
    '''
    return cv.putText(img, str(st), org, font, scale, color)

def vprint(msg, verbose=True, newline=True):
    '''
    Print depending on verbose option or not
    '''
    if verbose:
        if newline:
            print(msg)
        else:
            print(msg, end='')

def random_rectangle(space):
    '''
    Returns a random rectangle inside space
    Space is (width, height) 
    Return format: x, y, w, h
    '''    
    assert len(space) == 2, "random rectangles can only be created in 2D space, got {}D space".format(len(space))

    # Random "mask": height, width, top left row, top left colunm
    w = randint(1, space[0] - 1)
    h = randint(1, space[1] - 1)
    
    x = randint(0, space[0] - w)
    y = randint(0, space[1] - h)

    return x, y, w, h

def get_device(verbose=True):
    '''
    Gets pytorch current available device
    '''
    print("CUDA device available? ", end='')
    if torch.cuda.is_available() is True:
        device = torch.device("cuda:0")
        print("yes: " + str(device))
        print("GPU:" + str(get_device_name(0)))
    else:
        device = torch.device("cpu")
        print("no. Using CPU")
    return device

def parse_argv(argv):
    '''
    Returns bools representing what is going to be executed by main.py
    '''
    finetune = False
    display_volume = False
    train = True
    volume = False
    db_name = "clarissa"
    notest = False
    study_ths = False
    wait = 1

    valid_dbs = ["clarissa", "adni", "cc359", "concat"]

    if len(argv) > 1:
        if "notrain" in argv:
            print("skipping training")
            train = False
        if "volume" in argv:
            print("Volume test")
            volume = True
        if "display" in argv:
            print("Displaying brains")
            display_volume = True
        if "db" in argv:
            db_name = argv[argv.index("db") + 1]
            if db_name not in valid_dbs:
                raise ValueError("Invalid db identifier {}".format(db_name))
            print("Using db {}".format(db_name))
        if "notest" in argv:
            print("Not testing")
            notest = True
        if "wait" in argv:
            wait = int(argv[argv.index("wait") + 1])
            print("Custom display wait: {}".format(wait))
        if "ths" in argv:
            study_ths = True
            print("Studying THS")
        if "save" in argv:
            basepath = argv[argv.index("save") + 1]
            print("Changing basepath")
            if not os.path.exists(basepath):
                raise OSError("Given basepath {} does not exists".format(basepath))
        if "finetune" in argv:
            finetune = True
            print("Performing finetune, basename model will be loaded before training on specified db")

    return display_volume, train, volume, db_name, notest, study_ths, wait, finetune

'''
Adapted from ia636 toolbox
'''
def iarot90(img, axis='X'):
    PIVAL = math.pi
    g = 0

    if axis == 'X':
        Trx = np.array([[np.cos(PIVAL/2), -np.sin(PIVAL/2), 0, img.shape[1] - 1],
                    [np.sin(PIVAL/2),  np.cos(PIVAL/2), 0, 0],
                    [           0,             0, 1, 0],
                    [           0,             0, 0, 1]]).astype(float)
        g = iaffine(img, Trx, [img.shape[1], img.shape[0], img.shape[2]])

    elif axis == 'Y':
        Try = np.array([[ np.cos(PIVAL/2), 0, np.sin(PIVAL/2), 0],
                        [            0, 1,            0, 0],
                        [-np.sin(PIVAL/2), 0, np.cos(PIVAL/2), img.shape[0] - 1],
                        [            0, 0,            0, 1]])
        g = iaffine(img, Try, [img.shape[2], img.shape[1], img.shape[0]])
    elif axis == 'Z':
        Trz = np.array([[1,            0,             0, 0],
                        [0, np.cos(PIVAL/2), -np.sin(PIVAL/2), img.shape[2] - 1],
                        [0, np.sin(PIVAL/2),  np.cos(PIVAL/2), 0],
                        [0,            0,             0, 1]]).astype(float)
        g = iaffine(img, Trz, [img.shape[0], img.shape[2], img.shape[1]])

    return g

def iaffine(f,T,domain=0):

    if np.sum(domain) == 0:
        domain = f.shape
    if len(f.shape) == 2:
        H,W = f.shape
        y1,x1 = np.indices((domain))

        yx1 = np.array([ y1.ravel(),
                            x1.ravel(),
                            np.ones(np.product(domain))])
        yx_float = np.dot(np.linalg.inv(T), yx1)
        yy = np.rint(yx_float[0]).astype(int)
        xx = np.rint(yx_float[1]).astype(int)

        y = np.clip(yy,0,H-1).reshape(domain)
        x = np.clip(xx,0,W-1).reshape(domain)

        g = f[y,x]

    if len(f.shape) == 3:
        D,H,W = f.shape
        z1,y1,x1 = np.indices((domain))
        zyx1 = np.array([ z1.ravel(),
                            y1.ravel(),
                            x1.ravel(),
                            np.ones(np.product(domain)) ])
        zyx_float = np.dot(np.linalg.inv(T), zyx1)

        zz = np.rint(zyx_float[0]).astype(int)
        yy = np.rint(zyx_float[1]).astype(int)
        xx = np.rint(zyx_float[2]).astype(int)

        z = np.clip(zz, 0, D-1).reshape(domain) #z
        y = np.clip(yy, 0, H-1).reshape(domain) #rows
        x = np.clip(xx, 0, W-1).reshape(domain) #columns

        g = f[z,y,x]

    return g

