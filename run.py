'''
Module with functions to run subprocesses and manage files
Most of this were run only once
'''
from sys import argv
import traceback
import time
import os
import shutil
import glob
import subprocess
from tqdm import tqdm
import nibabel as nib
import cv2 as cv
import numpy as np
from utils import viewnii, normalizeMri, file_dialog, alert_dialog, confirm_dialog, error_dialog
from unet import UNet
from dunet import get_dunet
import torch
from dataset import orientations, mni_adni, default_adni
from transforms import run_once

PRE_REGISTER_VOL_PATH = 'cache/pre_register_vol.nii.gz'
PRE_REGISTER_MASK_PATH = 'cache/pre_register_mask.nii.gz'
MNI_BUFFER_VOL_PATH  = 'cache/mnibuffer.nii.gz'
MNI_BUFFER_MASK_PATH = 'cache/mnimaskbuffer.nii.gz'
MNI_BUFFER_MATRIX_PATH = 'cache/mnibuffer.mat'
INVERSE_MATRIX_PATH = 'cache/invmnibuffer.mat'
TEMP_MASK_PATH = 'cache/mask.nii.gz'
MASKS_FOLDER = "e2dhipseg_masks"

def get_models(bias, e2d, res, small, bn, dunet, dim='2d', model_folder=None, verbose=True):
    '''
    Navigates through past results folder to load a past result
    '''
    if dim == '3d':
        model = UNet(1 + (e2d*2), 1, residual=res, small=small, bias=bias, bn=bn, dim=dim, verbose=verbose)
        if model_folder is not None:
            path = glob.glob(os.path.join(model_folder, "*.pt"))[0]
            model.load_state_dict(torch.load(path))
        return model
    else:
        models = {}
        for o in orientations:
            if dunet:
                model = get_dunet()
            else:
                model = UNet(1 + (e2d*2), 1, residual=res, small=small, bias=bias, bn=bn, dim=dim, verbose=verbose)
            if model_folder is not None:
                path = glob.glob(os.path.join(model_folder, "*" + o + ".pt"))[0]
                model.load_state_dict(torch.load(path))
            models[o] = model
        return models

def hippodeep(folder="/home/diedre/git/hippodeep", display=True):
    '''
    Folder is where to look for .nii.gz files to run hippodeep
    '''
    for f in tqdm(glob.glob(os.path.join(folder, "*.nii.gz"))):
        try:
            int(os.path.basename(f).split(".nii.gz")[0])
            print(f)
        except ValueError:
            continue
        
        shellscript = subprocess.Popen(["sh", "deepseg3.sh", os.path.basename(f)])
        returncode = shellscript.wait()
        if display:
            result = nib.load(f[:-7] + "_mask_L.nii.gz").get_fdata() + nib.load(f[:-7] + "_mask_R.nii.gz").get_fdata()
            result[result > 1] = 1
            viewnii(normalizeMri(nib.load(f).get_fdata()), result)
            cv.destroyAllWindows()

def dcm2niix(folder):
    '''
    Runs external dcm2niix utility on given folder
    '''
    shellscript = subprocess.run(["/home/diedre/Downloads/NITRC-mricrogl-Downloads/mricrogl_linux/mricrogl_lx/dcm2niix", folder])

def nii2niigz(folder):
    '''
    Converts nii files in folder to nii.gz
    '''
    for f in tqdm(glob.iglob(os.path.join(folder, "*.nii"))):
        vol = nib.load(f)
        nib.save(vol, f + ".gz")

def mni152reg(invol, mask=None, ref_brain="/usr/local/fsl/data/standard/MNI152lin_T1_1mm.nii.gz", save_path=MNI_BUFFER_VOL_PATH, mask_save_path=MNI_BUFFER_MASK_PATH, remove=True, return_numpy=True, keep_matrix=False):
    '''
    Register a sample and (optionally) a mask from disk and returns them as numpy volumes
    '''
    try:
        ret = None
        shellscript = subprocess.run(["flirt", "-in",  invol, "-ref", ref_brain, "-out", save_path, "-omat", MNI_BUFFER_MATRIX_PATH])
        if return_numpy:
            vol = nib.load(MNI_BUFFER_VOL_PATH).get_fdata()
        
        if mask is None and return_numpy:
            ret = vol
        else:
            shellscript = subprocess.run(["flirt", "-in",  mask, "-ref", ref_brain, "-out", mask_save_path, "-init", MNI_BUFFER_MATRIX_PATH, "-applyxfm"])
            if return_numpy:
                mask = nib.load(MNI_BUFFER_MASK_PATH).get_fdata()
                ret = (vol, mask)
        
        if remove:
            try:
                os.remove(MNI_BUFFER_VOL_PATH)
                if not keep_matrix:
                    os.remove(MNI_BUFFER_MATRIX_PATH)
                if mask is not None:
                    os.remove(MNI_BUFFER_MASK_PATH)
            except OSError as oe:
                print("Error trying to remove mni register buffer files: {}".format(oe))
    except FileNotFoundError as fnfe:
        error_dialog("FLIRT registration error or FLIRT installation not found. Make sure FLIRT is installed for your OS.")
        print("Registration ERROR: {}".format(fnfe))
        quit()

    return ret

def invert_matrix(hip_path, ref_path, saved_matrix):
    '''
    Inverts the FSL matrix and returns hip to original state
    Returns path of final result
    '''
    print("Inverting matrix... {}".format(hip_path))
    shellscript = subprocess.run(["convert_xfm", "-omat",  INVERSE_MATRIX_PATH, "-inverse", saved_matrix])
    print("Transforming back to original space...")
    shellscript = subprocess.run(["flirt", "-in",  hip_path, "-ref", ref_path, "-out", "final_buffer.nii.gz", "-init", INVERSE_MATRIX_PATH, "-applyxfm"])
    
    save_path = ref_path + "_voxelcount-{}_e2dhipmask.nii.gz".format(int(nib.load("final_buffer.nii.gz").get_fdata().sum()))
    
    try:
        os.rename("final_buffer.nii.gz", save_path)
        os.remove(saved_matrix)
        os.remove(INVERSE_MATRIX_PATH)
        os.remove(hip_path)
    except OSError as oe:
        print("Error trying to remove post register matrix: {}".format(oe))
        return

    print("Post-Registrations done.")
    return save_path

def adni_iso2mni(source_path=default_adni, save_path=mni_adni):
    '''
    Register all isometric ADNI volumes to MNI152
    '''
    print("Registering all adni volumes to mni152")
    source_masks_path = os.path.join(source_path, "masks")
    for s in tqdm(glob.glob(os.path.join(source_path, "samples", "*.nii.gz"))):
        mask_name = os.path.basename(s).split(".nii.gz")[0]
        mni152reg(s, os.path.join(source_masks_path, mask_name + ".nii.gz"), 
                  save_path=os.path.join(save_path, "samples",mask_name), mask_save_path=os.path.join(save_path, "masks", mask_name),
                  remove=False, return_numpy=False)

def reg_handler(vol, mask=None):
    '''
    Just calls the right registration processing function
    '''
    print("Registering input...")
    if mask is None:
        return reg_pre_post_single(vol)
    else:
        return reg_pre_post_pair(vol, mask)

def reg_pre_post_pair(vol, mask):
    '''
    Pre and post processing of input volume and mask paths for mni152reg
    '''
    begin = time.time()
    if type(vol) == np.ndarray and type(mask) == np.ndarray:
        nib.save(nib.Nifti1Image(vol, affine=None), PRE_REGISTER_VOL_PATH)
        nib.save(nib.Nifti1Image(mask, affine=None), PRE_REGISTER_MASK_PATH)
        vol = PRE_REGISTER_VOL_PATH
        mask = PRE_REGISTER_MASK_PATH
    elif not(type(vol) == str) and not(type(mask) == str):
        raise ValueError("vol and mask should be a numpy volume or a path to the volume")

    print("Input PATHS -> Vol: {}\nMask: {}".format(vol, mask))
    vol, mask = mni152reg(vol, mask=mask, keep_matrix=True)
    
    if vol.max() > 1.0 or vol.min() < 0 or mask.max() > 1.0 or mask.min < 0:    
        print("Data out of range, normalizing...")    
        vol = normalizeMri(vol.astype(np.float32)).squeeze()
        mask = mask.astype(np.bool).astype(np.float32).squeeze()
    print("Registration took {}s".format(time.time() - begin))
    return vol, mask

def reg_pre_post_single(vol):
    '''
    Pre and post processing of input volume for mni152reg
    '''
    begin = time.time()
    # We want vol to be a path, but it can not be
    if type(vol) == np.ndarray:
        nib.save(nib.Nifti1Image(vol, affine=None), PRE_REGISTER_VOL_PATH)
        vol = PRE_REGISTER_VOL_PATH
    elif not(type(vol) == str):
        raise ValueError("vol should be a numpy volume or a path to the volume")

    print("Input PATH -> Vol: {}".format(vol))
    vol = mni152reg(vol, mask=None, keep_matrix=True)
    
    if vol.max() > 1.0 or vol.min() < 0:    
        print("Data out of range, normalizing...")    
        vol = normalizeMri(vol.astype(np.float32)).squeeze()
    print("Registration took {}s".format(time.time() - begin))
    return vol

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def main(runlist, models, reg, batch_mode, results_dst):
    '''
    Main loop to run externally
    '''
    for arg in tqdm(runlist):
        try:
            print("Processing {}\n".format(arg))
            if reg:
                # Register
                print("Performing MNI152 registration with FSL FLIRT due to -reg arg... This might take a few seconds...")
                volpath = arg
                arg = reg_handler(arg)

                # Segment
                vol, mask = run_once(None, models, numpy_input=arg, save=False)
                
                # Register back
                nib.save(nib.Nifti1Image(mask, affine=None), TEMP_MASK_PATH)
                mask_path = invert_matrix(TEMP_MASK_PATH, volpath, MNI_BUFFER_MATRIX_PATH)
            else:
                vol, mask, mask_path = run_once(arg, models, save=True, return_mask_path=True)

            try:
                check_file = os.path.join(results_dst, os.path.basename(mask_path))
                print("Saving result to {}".format(check_file))
                if os.path.isfile(check_file):
                    print("WARNING: Results for this volume already exist, overwriting...")
                    os.remove(check_file)
                shutil.move(mask_path, results_dst)
            except OSError as oe:
                msg = "Error moving results: {}, do you have the right permissions?".format(oe)
                print(msg)
                error_dialog(msg)
                quit()

            if "-display" in argv:
                print("Displaying results.")
                viewnii(vol, mask=mask)
        except Exception as e:
            traceback.print_exc()
            print("Error: {}, make sure your data is ok, and you have proper permissions. Please contact author https://github.com/dscarmo for issues".format(e))
            if batch_mode: print("Trying to continue... There might be errors in following segmentations.")

if __name__ == "__main__":
    mask = None
    try:
        arg = argv[1]
        folder = "/home/diedre/git/hippodeep"
        if len(argv) >= 4:
            mask = argv[3]
        if arg != "hippodeep" and len(argv) >= 3:
            folder = argv[2]
    except:
        arg = "run"

    if arg == "hippodeep":
        print("Due to author limitations, hippodeep must be run with terminal on the hippodeep folder, with the files on the same folder")
        hippodeep(folder)
    elif arg == "dcm2niix":
        dcm2niix(folder)
    elif arg == "nii2niigz":
        nii2niigz(folder)
    elif arg == "mni152reg":
        print("Vol: {}\nMask: {}".format(folder, mask))
        vol, mask = reg_handler(folder, mask)
        viewnii(vol, mask, wait=0)
    elif arg == "adni2mni":
        adni_iso2mni()
    else:
        batch_mode = False
        runlist = []
        if len(argv) == 1:
            alert_dialog("Please give a nift volume to run on.")
            arg = file_dialog()
            if arg is None:
                alert_dialog("No volume given, quitting.")
                quit()
            results_dst = os.path.join(os.path.dirname(arg), MASKS_FOLDER)    
            os.makedirs(results_dst, exist_ok=True)
            print("Results will be in {}\n".format(os.path.join(arg, MASKS_FOLDER)))
            reg = confirm_dialog("Do you want to register the volume to MNI152 space? (recommended, can take a few seconds)")
        else:
            if "-dir" in argv:
                assert os.path.isdir(arg), "folder given in -d argument is not a folder!"
                results_dst = os.path.join(arg, MASKS_FOLDER)    
                batch_mode = True
            else:
                results_dst = os.path.join(os.path.dirname(arg), MASKS_FOLDER)    
                assert os.path.isfile(arg), "File not found. Make sure the path for your nii input volume {} is correct. If its a directory use -dir".format(arg)

            os.makedirs(results_dst, exist_ok=True)
            print("Results will be in {}\n".format(os.path.join(arg, MASKS_FOLDER)))

            reg = "-reg" in argv
            print("Running pre-saved weights best model in {}".format(arg))

        models = get_models(False, True, True, False, True, False, dim='2d', model_folder="weights", verbose=False)

        if batch_mode:
            runlist = glob.glob(os.path.join(arg, "*.nii")) + glob.glob(os.path.join(arg, "*.nii.gz"))
            print("Running segmentation on the following files: {}\n".format(runlist))
            main(runlist, models, reg, batch_mode, results_dst)
        else:
            runlist.append(arg)
            main(runlist, models, reg, batch_mode, results_dst)
