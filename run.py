'''
Module with functions to run subprocesses and manage files
'''
from sys import argv
import os
import glob
import subprocess
from tqdm import tqdm
import nibabel as nib

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

def fancontrol():
    try:
        print("-------------------- Running nfancurve ------------------")
        shellscript = subprocess.run(["/home/diedre/git/nfancurve/update.sh"])
        print("-------------------- Done  ------------------")
    except Exception as e:
        print("Failed to control GPU fan: {}, continuing.".format(e))

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
        if arg != "hippodeep":
            folder = argv[2]
    except:
        print("Please give arg: which subprocess to run and which folder eg. python3 run.py nii2niigz /home/me/data")
    
    print("Running {} in {}".format(arg, folder))

    if arg == "hippodeep":
        print("Due to author limitations, hippodeep must be run with terminal on the hippodeep folder, with the files on the same folder")
        hippodeep(folder)
    elif arg == "dcm2niix":
        dcm2niix(folder)
    elif arg == "nii2niigz":
        nii2niigz(folder)
    elif arg == "mni152reg":
        mni152reg(folder, mask=mask)
    elif arg == "louder":
        fancontrol()
            
    
        

