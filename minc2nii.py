'''
This is work in progress trying to decifer simone's data.
MODULE IN DEVELOPMENT

Author: Diedre Carmo
https://github.com/dscarmo
'''
import os
from math import pi
from sys import argv
from glob import glob
import numpy as np
import nibabel as nib
import cv2 as cv
from utils import viewnii, tag2numpy, iarot90
from scipy.ndimage import rotate as rotate3d
from tqdm import tqdm
import numpy as np
import math
from multiprocessing import Process, Queue
from threed import plot_3d


def main(path=None):
    '''
    Tries to convert minc files in path folder
    Path should have folder all (source) and nift (destination)
    '''
    if path is None:
        #data_path = "/home/diedre/bigdata/simone/aline"
        data_path = "/home/diedre/bigdata/simone/all"
        print("Using default data path {}".format(data_path))
    else: 
        data_path = path
        print("Searching for minc files in {}".format(data_path))

    files = glob(data_path + "/*.mnc")
    dest_path = os.path.join(data_path, "nift")

    for f in files:
        # Pegar mascaras baseado no nome do arquivo minc
        fid = os.path.basename(f).split('.')[0]
        print(fid)
        emask_fname = os.path.join(data_path, fid + "e.tag")
        e_mask_fname = os.path.join(data_path, fid + "-e.tag")
        dmask_fname = os.path.join(data_path, fid + "d.tag")
        d_mask_fname = os.path.join(data_path, fid + "-d.tag")
        print(f)
        print(e_mask_fname)
        print(d_mask_fname)
        # Tenta com e sem o -
        try:
            etag = np.genfromtxt(emask_fname, missing_values=';', skip_header = 4)
        except OSError:
            etag = np.genfromtxt(e_mask_fname, missing_values=';', skip_header = 4)
        try:
            dtag = np.genfromtxt(dmask_fname, missing_values=';', skip_header = 4)
        except OSError:
            dtag = np.genfromtxt(d_mask_fname, missing_values=';', skip_header = 4)            
        
        # Carregar e normalizar volume, pegar mascaras e somar direita esquerda
        minc = nib.load(f)
        # Default affine
        '''affine = np.array([[0, 0, 1, 0],
                   [0, 1, 0, 0],
                   [1, 0, 0, 0],
                   [0, 0, 0, 1]])'''
        nii = nib.Nifti1Image(minc.get_data(), affine=minc.affine)
        
        data = nii.get_data()
        shape = data.shape

        # William
        '''vol_T1 = np.swapaxes(data, 0, -1)
        vol_T1 = np.swapaxes(vol_T1, 0, 1)
        vol_T1 = vol_T1[::-1]
        
        emask = tag2numpy(shape, etag)
        dmask = tag2numpy(shape, dtag)
        mask = emask + dmask
        
        vol, mask = np.rot90(vol_T1), np.rot90(mask)'''
        #
        
        # Danilo
        emask = tag2numpy(shape, etag)
        dmask = tag2numpy(shape, dtag)
        mask = emask + dmask
        vol = np.zeros((shape), dtype=data.dtype)
        cv.normalize(data, vol, alpha=1.0, beta=0.0, norm_type=cv.NORM_MINMAX)

        # Alinhar
        # TODO alignment is wrong
        #vol = iarot90(vol, axis = 'X')
        #vol = iarot90(vol, axis = 'Z')
        #vol = iarot90(vol, axis = 'Z')

        #mask = iarot90(mask, axis = 'X')
        #

        # Mostrar
        viewnii(vol, mask=mask, id="simone data", wait=50, rotate=0, fx=2, fy=2)
        
        
if __name__ == "__main__":
    if len(argv) > 1:
        main(argv[1])
    else:
        main()