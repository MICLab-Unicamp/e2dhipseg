'''
This module is related to generating H5PY dataset files for later use.

Author: Diedre Carmo
https://github.com/dscarmo
'''
import os
import numpy as np
import h5py
import nibabel as nib
import cv2
import glob
from sys import argv
import multiprocessing as mp 
from tqdm import tqdm
from matplotlib import pyplot as plt
from nibabel.testing import data_path 
from utils import normalizeMri, myrotate
from dataset import default_adni, mni_adni

orientations = ["sagital", "coronal", "axial"]
folders = ["samples", "masks"]

def normalizeForView(data):
    '''
    OLD VERSION KEPT FOR COMPATIBILITY SAKE
    Normalizes float content on MRI samples to 0-1, using min-max normalization
    
    data: input 3D numpy data do be normalized

    returns: normalized 3D numpy data 
    '''
    img = np.zeros((data.shape), dtype=np.float32)
    cv2.normalize(data, img, alpha=1.0, beta=0.0, norm_type=cv2.NORM_MINMAX)
    return img

def generate_h5py(filename="mni_hip_data_", path=os.path.join("/home", "diedre", "bigdata", "mni_hip_data"), hiponly=True, adni=False):
    '''
    Most modern dataset creation method, to keep float information
    '''
    print("Hiponly? {}".format("yes" if hiponly else "no"))
    print("ADNI? {}".format("yes" if adni else "no"))
    
    if hiponly:
        filename += "hiponly.hdf5"
    else:
        filename += "full.hdf5"

    ignored_slices = 0
    with h5py.File(os.path.join(path, filename), "w") as h5:
        for f in folders:
            h5.create_group(f)
            for o in orientations:
                h5[f].create_group(o)
        
        samples = glob.glob(os.path.join(path, "samples", "*.nii.gz"))
        
        for s in tqdm(samples):
            sample = nib.load(s)
            data = sample.get_fdata()

            fname = os.path.basename(s) 
            fid = fname[:-4]
            if adni:
                fid = fname.split('.')[0]
            if adni:
                data = normalizeMri(data).astype(np.float16)
            else:
                data = normalizeForView(data).astype(np.float16)
            
            mask = nib.load(os.path.join(path, "masks", fname))
            mdata = mask.get_fdata()
            
            if adni:
                mdata = mdata.astype(np.bool).astype(np.float16)
            else:
                mdata = (np.ma.masked_outside(mdata, 11.0, 12.0).filled(0) > 0).astype(np.float32)
                mdata = normalizeForView(mdata).astype(np.float16)

            shape = data.shape
            for o in orientations:
                for i in range(shape[orientations.index(o)]):
                    if o == 'sagital':
                        saveimg = data[i, :, :]
                        savemask = mdata[i, :, :]
                    elif o == 'coronal':
                        saveimg = data[:, i, :]
                        savemask = mdata[:, i, :]
                    elif o == 'axial':
                        saveimg = data[:, :, i]
                        savemask = mdata[:, :, i]
                    else:
                        raise ValueError("View has to be sagital, coronal or axial")

                    if hiponly is True:
                        if np.max(savemask) <= 0:
                            ignored_slices += 1
                            continue

                    saveimg = myrotate(saveimg, 90)
                    savemask = myrotate(savemask, 90)

                    si = '0'*(3-len(str(i))) + str(i)

                    h5["samples"][o].create_dataset(fid + "_" + si, data=saveimg, dtype=np.float16)
                    h5["masks"][o].create_dataset(fid + "_" + si, data=savemask, dtype=np.float16)
            

def vis_h5py(filename="mni_hip_data_", path=os.path.join("/home", "diedre", "bigdata", "mni_hip_data"), hiponly=True):
    print("Hiponly {}".format(hiponly))
    if hiponly:
        filename += "hiponly.hdf5"
    else:
        filename += "full.hdf5"
    with h5py.File(os.path.join(path, filename), "r") as h5:
        for o in orientations:
            for s in h5["samples"][o]:
                sample = h5["samples"][o].get(s)[:].astype(np.float)
                mask = h5["masks"][o].get(s)[:].astype(np.float)
                cv2.imshow("sample", sample)
                cv2.imshow("mask", mask)
                sh = sample.shape
                view3 = (sample + mask)/2
                view = np.hstack((sample, mask, view3))
                cv2.putText(view, str(o) + s + str(sh), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                cv2.imshow("Brain demo" + o, view)
                if cv2.waitKey(10) == 27:
                    break

if __name__ == "__main__":
    if len(argv) > 1:
        arg = argv[1]
        try:
            arg2 = argv[2]
        except:
            arg2 = None
            pass
        if arg == "vis":
            vis_h5py(hiponly=arg2=="hiponly")
        elif arg == "genh5":
            generate_h5py(hiponly=arg2=="hiponly") 
        elif arg == "genadni":
            generate_h5py(filename="float16_adni_hip_data_", path=default_adni, hiponly=True, adni=True)
        elif arg == "genmniadni":
            generate_h5py(filename="float16_mniadni_hip_data_", path=mni_adni, hiponly=True, adni=True)  
        elif arg == "visadni":
            vis_h5py(filename="float16_adni_hip_data_", path=default_adni, hiponly=True)
