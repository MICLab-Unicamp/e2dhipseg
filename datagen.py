'''
This module is related to generating H5PY dataset files for later use.

Author: Diedre Carmo
https://github.com/dscarmo
'''
import os
import glob
from sys import argv

import numpy as np
import h5py
import nibabel as nib
from tqdm import tqdm
import cv2 as cv

from utils import normalizeMri, myrotate
from dataset import default_adni, mni_adni, default_harp

ORIENTATIONS = ["sagital", "coronal", "axial"]
FOLDERS = ["samples", "masks"]


def normalize_for_view(data):
    '''
    OLD VERSION KEPT FOR COMPATIBILITY SAKE
    Normalizes float content on MRI samples to 0-1, using min-max normalization
    data: input 3D numpy data do be normalized
    returns: normalized 3D numpy data
    '''
    img = np.zeros((data.shape), dtype=np.float32)
    cv.normalize(data, img, alpha=1.0, beta=0.0, norm_type=cv.NORM_MINMAX)
    return img


def generate_h5py(filename="mni_hip_data_", path=os.path.join("/home", "diedre", "bigdata", "mni_hip_data"),
                  hiponly=True, adni=False, harp=False):
    '''
    Most modern dataset creation method, to keep float information
    '''
    print("Hiponly? {}".format("yes" if hiponly else "no"))
    print("ADNI? {}".format("yes" if adni else "no"))
    print("HARP? {}".format("yes" if harp else "no"))
    assert harp != adni, " gen h5py arguments conflicting"

    if hiponly:
        filename += "hiponly.hdf5"
    else:
        filename += "full.hdf5"

    ignored_slices = 0
    with h5py.File(os.path.join(path, filename), "w") as h5:
        for f in FOLDERS:
            h5.create_group(f)
            for o in ORIENTATIONS:
                h5[f].create_group(o)
        if harp:
            samples = list(glob.iglob(os.path.join(path, "samples", "**", "*.nii.gz"), recursive=True))
        else:
            samples = glob.glob(os.path.join(path, "samples", "*.nii.gz"))

        for s in tqdm(samples):
            sample = nib.load(s)
            data = sample.get_fdata()

            fname = os.path.basename(s)
            fid = fname[:-4]
            if adni or harp:
                fid = fname.split('.')[0]
            if adni or harp:
                data = normalizeMri(data).astype(np.float16)
            else:
                data = normalize_for_view(data).astype(np.float16)

            if harp:
                mask = nib.load(os.path.join(path, "masks", "all", fname))
            else:
                mask = nib.load(os.path.join(path, "masks", fname))
            mdata = mask.get_fdata()

            if adni or harp:
                mdata = mdata.astype(np.bool).astype(np.float16)
            else:
                mdata = (np.ma.masked_outside(mdata, 11.0, 12.0).filled(0) > 0).astype(np.float32)
                mdata = normalize_for_view(mdata).astype(np.float16)

            shape = data.shape
            for o in ORIENTATIONS:
                for i in range(shape[ORIENTATIONS.index(o)]):
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

                    separator = '-' if harp else '_'

                    h5["samples"][o].create_dataset(fid + separator + si, data=saveimg, dtype=np.float16)
                    h5["masks"][o].create_dataset(fid + separator + si, data=savemask, dtype=np.float16)


def vis_h5py(filename="mni_hip_data_", path=os.path.join("/home", "diedre", "bigdata", "mni_hip_data"), hiponly=True):
    print("Hiponly {}".format(hiponly))
    if hiponly:
        filename += "hiponly.hdf5"
    else:
        filename += "full.hdf5"
    with h5py.File(os.path.join(path, filename), "r") as h5:
        for o in ORIENTATIONS:
            for s in h5["samples"][o]:
                sample = h5["samples"][o].get(s)[:].astype(np.float)
                mask = h5["masks"][o].get(s)[:].astype(np.float)
                cv.imshow("sample", sample)
                cv.imshow("mask", mask)
                sh = sample.shape
                view3 = (sample + mask)/2
                view = np.hstack((sample, mask, view3))
                cv.putText(view, str(o) + s + str(sh), (20, 20), cv.FONT_HERSHEY_SIMPLEX, 1, 1)
                cv.imshow("Brain demo" + o, view)
                if cv.waitKey(10) == 27:
                    break


if __name__ == "__main__":
    if len(argv) > 1:
        arg = argv[1]
        try:
            arg2 = argv[2]
        except IndexError:
            arg2 = None
            pass
        if arg == "vis":
            vis_h5py(hiponly=(arg2 == "hiponly"))
        elif arg == "genh5":
            generate_h5py(hiponly=(arg2 == "hiponly"))
        elif arg == "genadni":
            generate_h5py(filename="float16_adni_hip_data_", path=default_adni,
                          hiponly=True, adni=True, harp=False)
        elif arg == "genmniadni":
            generate_h5py(filename="float16_mniadni_hip_data_", path=mni_adni,
                          hiponly=True, adni=True, harp=False)
        elif arg == "genharp":
            generate_h5py(filename="float16_harp_hip_data_", path=default_harp,
                          hiponly=True, adni=False, harp=True)
        elif arg == "visadni":
            vis_h5py(filename="float16_adni_hip_data_", path=default_adni,
                     hiponly=True)
