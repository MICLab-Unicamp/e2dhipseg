'''
Utility to view nift files in a given folder (argument)
Also performs adjustments to ADNI files
'''
import os
from os.path import join as add_path
import numpy as np
import nibabel as nib
from glob import iglob
from sys import argv
from utils import viewnii, normalizeMri, file_dialog, confirm_dialog, alert_dialog
from scipy.ndimage import rotate as rotate3d
from run import mni152reg


def ADNI_viewer(adjust=False, mniregister=False):
    '''
    Walks through ADNI folder checking things
    .nii files are originals
    .gz already adjusted (rotated)
    '''
    assert not(adjust and mniregister), " cant do both adjust and mniregister in adni"
    folder = "/home/diedre/bigdata/manual_selection_rotated/ADNI"
    save_folder = "/home/diedre/bigdata/manual_selection_rotated/raw2mni"
    print("Looking into {}".format(folder))
    n = 0
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
        elif mniregister:
            n += 1
            print("Registering...")
            mni152reg(volpath, hippath, save_path=os.path.join(save_folder, "samples", str(n) + ".nii.gz"),
                      mask_save_path=os.path.join(save_folder, "masks", str(n) + ".nii.gz"), remove=False, return_numpy=False)
        else:
            generic_viewer(volpath, hippath, print_header=False)


def ADNI_adjust(vol1, vol2):
    '''
    Adjusts vol to mask orientation (rotations)
    '''
    if vol1[-4:] != ".nii" or vol2[-4:] != ".nii":
        raise ValueError("Pls pass .nii original files to adjust")

    nii = nib.load(vol1)
    vol = nii.get_fdata()
    vol = rotate3d(vol, 90, (0, 2), order=0)
    vol = rotate3d(vol, 180, (1, 2), order=0)
    header = nii.get_header()

    nii2 = nib.load(vol2)
    header['pixdim'] = nii2.get_header()['pixdim']

    nib.save(nib.nifti1.Nifti1Image(vol, None, header=header), vol1[:-4] + ".nii.gz")
    nib.save(nii2, vol2[:-4] + ".nii.gz")


def generic_viewer(vol1, vol2=None, vol3=None, print_header=True, border_only=False):
    have_mask = vol2 is not None
    have_ref = vol3 is not None
    nii = nib.load(vol1)
    if vol1.split('.')[1] == 'npz':
        vol, mask = nii['vol'], nii['mask']
        viewnii(vol.astype(np.float32), mask=mask.astype(np.float32), border_only=border_only)
    else:
        vol = nii.get_fdata()
        norm_vol = normalizeMri(vol.astype(np.float32)).squeeze()
        print("vol1 shape: {}".format(norm_vol.shape))
        if print_header:
            print(nii.get_header())

        if have_ref and have_mask:
            hip = nib.load(vol2).get_fdata()
            hip2 = nib.load(vol3).get_fdata()
            norm_hip = hip.astype(np.bool).astype(np.float32).squeeze()
            norm_hip2 = hip2.astype(np.bool).astype(np.float32).squeeze()
            print("vol2 shape: {}".format(norm_hip.shape))
            print("vol3 shape: {}".format(norm_hip2.shape))
            viewnii(norm_vol, mask=norm_hip, ref=norm_hip2, border_only=border_only)
        if have_mask:
            hip = nib.load(vol2).get_fdata()
            norm_hip = hip.astype(np.bool).astype(np.float32).squeeze()
            print("vol2 shape: {}".format(norm_hip.shape))
            viewnii(norm_vol, mask=norm_hip, border_only=border_only)
        else:
            viewnii(norm_vol, border_only=border_only)


if __name__ == "__main__":
    nargs = len(argv)
    border_only = False
    if nargs == 1:
        vol = None
        mask = None
        ref = None
        alert_dialog("Please select a nift volume.")
        vol = file_dialog()
        if vol is not None:
            if confirm_dialog("Do you want to give a prediction mask to overlap?"):
                mask = file_dialog()
                if mask is None:
                    print("WARNING: No mask was given.")
                if confirm_dialog("Do you want to give a reference mask to overlap? Will be shown as the blue mask, while the"
                                  "prediction will be green."):
                    ref = file_dialog()
                    if ref is None:
                        print("WARNING: No reference was given.")
        else:
            print("No volume given to process, exiting.")
            quit()
        generic_viewer(vol, vol2=mask, vol3=ref, border_only=border_only)
    elif nargs == 2:
        if argv[1] == 'adni_adjust':
            ADNI_viewer(adjust=True)
        elif argv[1] == 'adni':
            ADNI_viewer()
        elif argv[1] == 'raw2mni':
            ADNI_viewer(mniregister=True)
        else:
            generic_viewer(argv[1], border_only=border_only)
    elif nargs == 3:
        generic_viewer(argv[1], vol2=argv[2], border_only=border_only)
    elif nargs == 4:
        generic_viewer(argv[1], vol2=argv[2], vol3=argv[3], border_only=border_only)
