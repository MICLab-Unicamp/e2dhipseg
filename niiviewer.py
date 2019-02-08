'''
Utility to view nift files in a given folder (argument)
Also performs adjustments to ADNI files
'''
import os
from os.path import join as add_path
import numpy as np
import cv2 as cv
import nibabel as nib
from glob import iglob
from sys import argv
from utils import viewnii, imagePrint, normalizeMri, myrotate
from scipy.ndimage import affine_transform, rotate
from shutil import copyfile

def ADNI_viewer(adjust=False):
    '''
    Walks through ADNI folder checking things
    .nii files are originals
    .gz already adjusted
    '''
    folder = argv[1] if len(argv) > 1 else "/home/diedre/bigdata/manual_selection_rotated/ADNI"
    print("Looking into {}".format(folder))
    for f in iglob(add_path(folder, "*")):
        for F in iglob(add_path(f, "**/*.nii" + (not adjust)*'.gz'), recursive=True):
            name = os.path.basename(F)
            print(name)
            tokens = name.split('_')
            if "Hippocampal" in tokens:
                hippath = F
            elif "MPR" or "MPR-R" in tokens:
                volpath = F

        if adjust:
            ADNI_adjust(volpath, hippath)
        else:
            generic_viewer(volpath, hippath, print_header=False)

def ADNI_adjust(vol1, vol2):
    '''
    Adjusts vol to mask orientation
    '''
    if vol1[-4:] != ".nii" or vol2[-4:] != ".nii":
        raise ValueError("Pls pass .nii original files to adjust")
    
    nii = nib.load(vol1)
    vol = nii.get_fdata()
    vol = rotate(vol, 90, (0, 2), order=0)
    vol = rotate(vol, 180, (1, 2), order=0)
    header = nii.get_header()
    
    nii2 = nib.load(vol2)
    header['pixdim'] = nii2.get_header()['pixdim']
    

    nib.save(nib.nifti1.Nifti1Image(vol, None, header=header), vol1[:-4] + ".nii.gz")
    nib.save(nii2, vol2[:-4] + ".nii.gz")


def generic_viewer(vol1, vol2=None, print_header=True):
    have_mask = vol2 != None
    nii = nib.load(vol1)
    vol = nii.get_fdata()
    norm_vol = normalizeMri(vol.astype(np.float32)).squeeze()
    print("vol1 shape: {}".format(norm_vol.shape))
    if print_header:
        print(nii.get_header())

    if have_mask:
        hip = nib.load(vol2).get_fdata()
        norm_hip = hip.astype(np.bool).astype(np.float32).squeeze()
        print("vol2 shape: {}".format(norm_hip.shape))
        viewnii(norm_vol, mask=norm_hip)
    else:
        viewnii(norm_vol)


if __name__ == "__main__":
    nargs = len(argv)
    if nargs == 1:
        ADNI_viewer(adjust=False)    
    elif nargs == 2:
        if argv[1] == 'adjust':
            ADNI_viewer(adjust=True)
        else:
            generic_viewer(argv[1])
    else:
        generic_viewer(argv[1], vol2=argv[2])        

