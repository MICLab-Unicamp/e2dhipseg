'''
Module with functions to run subprocesses and manage files
'''
from sys import argv
import os
import glob
import subprocess
from tqdm import tqdm
import nibabel as nib
from train import run_once
from utils import viewnii
from unet import UNet
from dunet import get_dunet
import torch
from dataset import orientations

def get_models(bias, e2d, res, small, bn, dunet, model_folder=None):
    '''
    Navigates through past results folder to load a past result
    WORKAROUND, DUPLICATED
    '''
    models = {}

    for o in orientations:
        if dunet:
            model = get_dunet()
        else:
            model = UNet(1 + (e2d*2), 1, residual=res, small=small, bias=bias, bn=bn, verbose=False)
        if model_folder is not None:
            path = glob.glob(os.path.join(model_folder, "*" + o + ".pt"))[0]
            model.load_state_dict(torch.load(path))
        models[o] = model
    return models

def hippodeep(folder="/home/diedre/git/hippodeep"):
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

def mni152reg(invol, mask=False, ref_brain="/usr/local/fsl/data/standard/MNI152lin_T1_1mm.nii.gz", ref_hip="/usr/local/fsl/data/mist/masks/hippocampus_thr75.nii.gz"):
    if mask:
        hip = "hip"
    else:
        hip = ''
    shellscript = subprocess.run(["flirt", "-in",  invol, "-ref", ref_brain, "-out", "mni/testmni" + hip, "-omat", "mni/testmni{}.mat".format(hip)])

if __name__ == "__main__":
    mask = None
    try:
        arg = argv[1]
        folder = "/home/diedre/git/hippodeep"
        if len(argv) >= 4:
            if argv[3] == "mask":
                mask = True
            else:
                mask = False
        if arg != "hippodeep" and len(argv) >= 3:
            folder = argv[2]
    except:
        print("Please give arg: which subprocess to run and which folder eg. python3 run.py nii2niigz /home/me/data")
    
    print("Running {}".format(arg))

    if arg == "hippodeep":
        print("Due to author limitations, hippodeep must be run with terminal on the hippodeep folder, with the files on the same folder")
        hippodeep(folder)
    elif arg == "dcm2niix":
        dcm2niix(folder)
    elif arg == "nii2niigz":
        nii2niigz(folder)
    elif arg == "mni152reg":
        mni152reg(folder, mask=mask)
    else:
        print("Running pre-saved weights best model in volume {}".format(arg))
        if not os.path.isfile(arg):
            raise OSError("File not found. Make sure the path for your nii input volume {} is correct".format(arg))
        models = get_models(False, True, True, False, True, False, "weights")
        vol, mask = run_once(arg, models)
        print("Done!")
        if "-nodisplay" not in argv:
            print("Displaying results, add argument -nodisplay if you dont want to check results.")
            viewnii(vol, mask=mask)
        