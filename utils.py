'''
Utility functions and constants that dont belong to a specific module

Author: Diedre Carmo
https://github.com/dscarmo
'''
from random import randint
import os
import subprocess
import math
import glob
import datetime
import numpy as np
import torch
import copy
import multiprocessing as mp
import psutil
from torch import Tensor
from typing import Set, Iterable
from torch.cuda import get_device_name
import cv2 as cv
import nibabel as nib
from matplotlib import pyplot as plt
import time
from skimage.transform import rotate
from scipy.sparse import coo_matrix
from scipy.ndimage import distance_transform_edt as distance
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox

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
D = 100

MULTI_TASK_NCHANNELS = 17
HALF_MULTI_TASK_NCHANNELS = 9
GDL_TWO = 2

multi_task_labels = [
    "other",  # 0
    "Left ventricle",  # 1
    "Right ventricle",  # 2
    "Left caudate",  # 3
    "Right caudate",  # 4
    "Left putamen",  # 5
    "Right putamen",  # 6
    "Left thalamus",  # 7
    "Right thalamus",  # 8
    "Left globus pallidus",  # 9
    "Right globus pallidus",  # 10
    "Left hipocampus",  # 11
    "Right hipocampus",  # 12
    "Left amigdala",  # 13
    "Right amigdala",  # 14
    "Left accumbens",  # 15
    "Right accumbens",  # 16
]

# No background on this labeling
half_multi_task_labels = [
    "other",  # 0
    "ventricle",  # 1
    "caudate",  # 2
    "putamen",  # 3
    "thalamus",  # 4
    "globus pallidus",  # 5
    "hipocampus",  # 6
    "amigdala",  # 7
    "accumbens",  # 8
]

limit_multi_labels = [
    "ventricle",  # 0
    "caudate",  # 1
    "putamen",  # 2
    "thalamus",  # 3
    "hipocampus",  # 4
]

# For display, BGR
color_map = {"ventricle": [0, 0, 1],  # red
             "caudate": [0, 1, 0],  # green
             "putamen": [1, 0, 0],  # blue
             "thalamus": [1, 1, 0],  # pool blue
             "globus pallidus": [1, 0, 1],  # pink
             "hipocampus": [0, 1, 1],  # yellow
             "amigdala": [0.3, 0, 0.3],  # dark purple
             "accumbens": [0, 0.5, 1]  # orange
             }


def split_l_r(volume, sagittal_axis=0):
    '''
    This assumes the brain is centered
    '''

    sagittal_len = volume.shape[sagittal_axis]
    internalv_l, internalv_r = copy.deepcopy(volume), copy.deepcopy(volume)

    if sagittal_axis == 0:
        internalv_r[:sagittal_len//2] = 0
        internalv_l[sagittal_len//2:] = 0
    elif sagittal_axis == 1:
        internalv_r[:, :sagittal_len//2] = 0
        internalv_l[:, sagittal_len//2:] = 0
    elif sagittal_axis == 2:
        internalv_r[:, :, :sagittal_len//2] = 0
        internalv_l[:, :, sagittal_len//2:] = 0
    else:
        raise ValueError("Invalid sagittal axis")

    return {"left": internalv_l, "right": internalv_r}


def cv_display_attention(att, display_engine='cv'):
    norm_att = (att - att.min()) / (att.max() - att.min())
    cv.imshow("Attention", cv.applyColorMap((norm_att*255).astype(np.uint8), cv.COLORMAP_HOT))
    norm_att = 1 - norm_att
    cv.imshow("Attention (inverted)", cv.applyColorMap((norm_att*255).astype(np.uint8), cv.COLORMAP_HOT))


# Simple GUI utils
def file_dialog():
    '''
    Simple GUI to chose files
    '''
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
    if filename:
        return filename
    else:  # if empty return None
        return None


def alert_dialog(msg, title="E2D HipSeg"):
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    messagebox.showinfo(title, msg)


def error_dialog(msg, title="E2D HipSeg"):
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    messagebox.showerror(title, msg)


def confirm_dialog(msg, title='E2D HipSeg'):
    '''
    Simple confirmation dialog
    '''
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    MsgBox = messagebox.askquestion(title, msg)
    if MsgBox == 'yes':
        return True
    else:
        return False


# Assert utils from github -> LIVIAETS/surface-loss
def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])
#


def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))

    b, w, h = seg.shape

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)

    return res


# switch between representations
def probs2class(probs: Tensor) -> Tensor:
    b, _, w, h = probs.shape
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, w, h)

    return res


def probs2one_hot(probs: Tensor) -> Tensor:
    _, C, _, _ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), C)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res


def tensor_statistics(t):
    print("Shape: {}, max: {}, min: {}, type: {}, sum: {}".format(t.shape, t.max(), t.min(), t.dtype, t.sum()))


def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    assert one_hot(torch.Tensor(seg), axis=0)
    C: int = len(seg)

    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res


def get_borders(img, kernel_size=3):
    '''
    Returns borders for 3D or 2D image with morphology
    '''
    ret = np.zeros_like(img)
    kernel = (kernel_size, kernel_size)
    if ret.ndim == 3:
        for i in range(ret.shape[0]):
            uint = (img[i]*255).astype(np.uint8)
            ret[i] = ((uint - cv.erode(uint, kernel, iterations=1))/255).astype(img.dtype)
    elif ret.ndim == 2:
        uint = (img*255).astype(np.uint8)
        ret = ((uint - cv.erode(uint, kernel, iterations=1))/255).astype(img.dtype)
    else:
        raise ValueError("input dimension not supported for border extraction")

    return ret


def get_memory_state():
    '''
    Returns memory information in gigabytes
    '''
    BILLION = 1000000000
    memory_dict = psutil.virtual_memory()
    memory_factor = 64//(memory_dict.total // BILLION)
    memory_state = {"free": memory_dict.free / BILLION,
                    "available": memory_dict.available / BILLION,
                    "total": memory_dict.total / BILLION,
                    "memory_factor":  1 if memory_factor == 0 else memory_factor}
    return memory_state


def type_assert(_type, *args):
    '''
    Asserts all args to be of type
    '''
    for arg in args:
        assert isinstance(arg, _type)


def get_slice(vol, i, orientation, rotate=90, orientations=orientations):
    ndim = vol.ndim
    if orientation not in orientations:
        raise ValueError("Unsupported orientation {} should be one of {}".format(orientation, orientations))
    ix = orientations.index(orientation)
    if ndim == 3:
        if ix == 0:
            return myrotate(vol[i, :, :], rotate)
        elif ix == 1:
            return myrotate(vol[:, i, :], rotate)
        elif ix == 2:
            return myrotate(vol[:, :, i], rotate)
    elif ndim == 4:
        if ix == 0:
            post_rotation_shape = myrotate(np.zeros(vol[0, 0, :, :].shape, dtype=np.uint8), rotate).shape
            ret_slice = np.zeros((vol.shape[0], post_rotation_shape[0], post_rotation_shape[1]), dtype=vol.dtype)
            for dim in range(vol.shape[0]):
                ret_slice[dim] = myrotate(vol[dim, i, :, :], rotate)
        elif ix == 1:
            post_rotation_shape = myrotate(np.zeros(vol[0, :, 0, :].shape, dtype=np.uint8), rotate).shape
            ret_slice = np.zeros((vol.shape[0], post_rotation_shape[0], post_rotation_shape[1]), dtype=vol.dtype)
            for dim in range(vol.shape[0]):
                ret_slice[dim] = myrotate(vol[dim, :, i, :], rotate)
        elif ix == 2:
            post_rotation_shape = myrotate(np.zeros(vol[0, :, :, 0].shape, dtype=np.uint8), rotate).shape
            ret_slice = np.zeros((vol.shape[0], post_rotation_shape[0], post_rotation_shape[1]), dtype=vol.dtype)
            for dim in range(vol.shape[0]):
                ret_slice[dim] = myrotate(vol[dim, :, :, i], rotate)
        return ret_slice
    else:
        raise ValueError("ndims for slice acquistion not supported")


def set_slice(vol, data, i, dim, add=False):
    '''
    Sets data orthogonally in a volume, replacing or adding
    '''
    vol_ndim = len(vol.shape)
    data_ndim = len(data.shape)
    assert dim < vol_ndim and data_ndim == vol_ndim - 1

    if dim == 0:
        if add:
            vol[i, :, :] += data
        else:
            vol[i, :, :] = data
    elif dim == 1:
        if add:
            vol[:, i, :] += data
        else:
            vol[:, i, :] = data
    elif dim == 2:
        if add:
            vol[:, :, i] += data
        else:
            vol[:, :, i] = data


def tag2numpy(shape, file_tag):
    '''
    Interpret tag file segmentation
    '''
    points = np.vstack([file_tag[:, :3]])

    pts_calculados = np.zeros(file_tag.shape).astype(int)
    pts_calculados[:, 0] = points[:, 0] + (shape[2] / 2)
    pts_calculados[:, 1] = points[:, 1] + (shape[0] / 2)
    pts_calculados[:, 2] = points[:, 2] + (shape[1] / 2)

    esquerda_direita = pts_calculados[:, 0]
    anterior_posterior = pts_calculados[:, 1]
    inferior_superior = pts_calculados[:, 2]

    seg_mask = np.zeros(shape).astype('bool')
    seg_mask[anterior_posterior, inferior_superior, esquerda_direita] = True

    return seg_mask


def plots(ths_study, cons, study_ths, opt="", savepath=None, mean_result=0, name="", std_result=0, rot=False, results=None):
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
        rot = "_random_rots" if rot else ''
        with open(os.path.join(savepath, opt + str(mean_result) + "meancons" + str(std_result) + "std" + rot + ".txt"), 'a') as f:
            f.write(str(cons))
            f.write("Final result: " + str(mean_result))
        if results is not None:
            with open(os.path.join(savepath, name + "eval_metrics.txt"), 'a') as f:
                f.write(results)
        plt.figure(num=name)
        plt.title("Individual VS Consensus DICE")
        plt.ylabel("DICE")
        plt.ylim(-0.1)
        labels = ["NP-Sagital", "Sagital", "NP-Coronal", "Coronal", "NP-Axial", "Axial", "NP-Consensus", "Consensus"]
        plt.boxplot(cons, labels=labels)
        plt.tight_layout()
        if savepath is not None:
            print("Saving consensus")
            plt.savefig(os.path.join(savepath, opt + "cons.eps"), format="eps", dpi=1000)
        else:
            print("Warning: Consensus not saved")


def keys2str(keys):
    '''
    Returns keys from a dict as a sane string
    '''
    ret = ''
    for i, key in enumerate(keys):
        lenkeys = len(keys)
        if i == lenkeys - 2:
            endchar = " and "
        elif i == lenkeys - 1:
            endchar = ''
        else:
            endchar = ", "

        ret += str(key) + endchar
    return ret


def check_min_max_type(min_limit, max_limit, _type_, verbose=True, **kwargs):
    '''
    Useful assert for min/max/type of numpy arrays
    '''
    if verbose:
        print("Checking min/max/type of {} with {}, {}, {}".format(keys2str(kwargs.keys()), min_limit, max_limit, _type_))
    for keyword, arg in kwargs.items():
        if verbose:
            print("{}: min {}, max {}, type {}".format(keyword, arg.min(), arg.max(), arg.dtype))
        if arg.max() <= max_limit and arg.min() >= min_limit and arg.dtype == _type_:
            continue
        else:
            return False
    return True


def normalizeMri(data):
    '''
    Normalizes float content on MRI samples to 0-1, using min-max normalization

    data: input 3D numpy data do be normalized
    returns: normalized 3D numpy data
    '''
    img = np.zeros((data.shape), dtype=data.dtype)
    cv.normalize(data, img, alpha=1.0, beta=0.0, norm_type=cv.NORM_MINMAX)
    return img


def cache_a_volume(vol, fid=None):
    '''
    Saves a volume as Nift file in cache folder
    Used for tools that need a path as input
    '''
    cache_path = os.path.join("cache", fid + str(datetime.datetime.now()) + ".nii.gz")
    nib.save(nib.Nifti1Image(vol, affine=None), cache_path)
    return cache_path


def clear_cache(*fids):
    '''
    Clears volumes in a fid array, or all cache volumes
    '''
    toremove = []
    if len(fids) == 0:
        for filename in glob.iglob(os.path.join("cache", "*")):
            toremove.append(filename)
    else:
        for fid in fids:
            for filename in glob.iglob(os.path.join("cache", "{}*".format(fid))):
                toremove.append(filename)

    for file_to_remove in toremove:
        try:
            os.remove(file_to_remove)
        except FileNotFoundError:
            print("Trying to remove unexistent file {} in cache.".format(filename))
        except Exception as e:
            print("Internal error in cache clearance: {}".format(e))


def spawn_itk(itk_manager, vol, mask=None, ref=None):
    '''
    Calls itksnap via its cli
    '''
    try:
        vol_id = "vol" + str(datetime.datetime.now())
        mask_id = "mask" + str(datetime.datetime.now())
        ref_id = "ref" + str(datetime.datetime.now())
        volpath = cache_a_volume(vol, vol_id)

        if mask is None and ref is None:
            status = subprocess.Popen(["bin/itksnap3.6/bin/itksnap",
                                       "-g", volpath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        elif ref is None:
            maskpath = cache_a_volume(mask, mask_id)
            status = subprocess.Popen(["bin/itksnap3.6/bin/itksnap",
                                       "-g", volpath,
                                       "-s", maskpath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            maskpath = cache_a_volume(mask, mask_id)
            refpath = cache_a_volume(ref, ref_id)
            status = subprocess.Popen(["bin/itksnap3.6/bin/itksnap",
                                       "-g", volpath, "-o", refpath,
                                       "-s", maskpath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        while status.poll() is None and not itk_manager.sig_term.value:
            time.sleep(1)

        if status.poll() is None:
            print("Closing lingering ITK windows...")
            status.terminate()
        clear_cache(vol_id, mask_id, ref_id)
        itk_manager.dealloc()
    except FileNotFoundError as fe:
        print("Error trying to use itksnap. Do you have it in the bin folder?: {}".format(fe))
    except Exception as e:
        print("Internal error in ITK Spawner: {}".format(e))


class ITKManager():
    '''
    Global class managing ITK spawns
    '''
    def __init__(self):
        self.itk_alloc_max = 2
        self.currently_alloced = mp.Manager().Value('i', 0)
        self.sig_term = mp.Manager().Value('i', 0)
        self.ps = []

    def increase_max(self):
        self.itk_alloc_max += 1

    def decrease_max(self):
        self.itk_alloc_max -= 1

    def alloc(self, args):
        while not self.available():
            cv.imshow("ITK-Manager Status", imagePrint(np.zeros((200, 1200)), str(self) + " Press ESC to quit."))
            key = cv.waitKey(1)
            if key == 27:
                return 1

        self.currently_alloced.value = self.currently_alloced.value + 1

        p = mp.Process(target=spawn_itk, args=(self,) + args)
        p.start()
        self.ps.append(p)

    def dealloc(self):
        self.currently_alloced.value = self.currently_alloced.value - 1

    def available(self):
        return self.currently_alloced.value < self.itk_alloc_max

    def terminate(self):
        self.sig_term.value = 1

        for p in self.ps:
            p.join()

    def __str__(self):
        return "{}/{} ITKs running.".format(self.currently_alloced.value, self.itk_alloc_max)


def viewnii(sample, mask=None, ref=None, id="", wait=1, rotate=0, mrotate=None, quit_on_esc=True, fx=2, fy=2, customshape=None,
            imprint=False, border_only=True, multi_labels=None, label=None, print_slice=True, itk_manager=None):
    '''
    Simple visualization of all orientations of mri
    '''
    print("Sample Shape: {} Max: {} Min: {} Type: {}".format(sample.shape, sample.max(), sample.min(), sample.dtype))
    if mask is not None:
        print("Mask Shape: {} Max: {} Min: {} Type: {}".format(mask.shape, mask.max(), mask.min(), mask.dtype))
    if itk_manager is not None:
        try:
            print("Using ITK-Snap for volume visualization.")
            ret_code = itk_manager.alloc((sample, mask, ref))
            if ret_code == 1:
                print("Captured CTRL-C, gracefully closing...")
                itk_manager.terminate()
                quit()
            else:
                return
        except Exception as e:
            print("Visualization internal error: {}".format(e))
            print("Falling back to simple visualization.")

    if customshape is None:
        shape = sample.shape
    else:
        shape = customshape

    if mrotate is None:
        mrotate = rotate
    else:
        print("Rotating mask by {}".format(mrotate))
    print("Rotating by {} for view".format(rotate))

    kup = True
    for j in range(0, 3):
        # for k in range(0, shape[j]):
        k = 0
        while k >= 0 and k < shape[j]:
            if j == 0:
                view = myrotate(sample[k, :, :], rotate)
                if mask is not None:
                    mview = myrotate(mask[k, :, :], mrotate)
                if ref is not None:
                    rview = myrotate(ref[k, :, :], mrotate)
            elif j == 1:
                view = myrotate(sample[:, k, :], rotate)
                if mask is not None:
                    mview = myrotate(mask[:, k, :], mrotate)
                if ref is not None:
                    rview = myrotate(ref[:, k, :], mrotate)
            elif j == 2:
                view = myrotate(sample[:, :, k], rotate)
                if mask is not None:
                    mview = myrotate(mask[:, :, k], mrotate)
                if ref is not None:
                    rview = myrotate(ref[:, :, k], mrotate)

            if border_only:
                kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
                if mask is not None:
                    uintmask = (mview*255).astype(np.uint8)
                    mask_borders = cv.Canny(uintmask, 10, 30)
                    mask_borders = cv.dilate(mask_borders, kernel)/255
                    mview = mview*mask_borders
                if ref is not None:
                    uintref = (rview*255).astype(np.uint8)
                    ref_borders = cv.Canny(uintref, 10, 30)
                    ref_borders = cv.dilate(ref_borders, kernel)/255
                    rview = rview*ref_borders

            view = cv.cvtColor(cv.resize(view, (0, 0), fx=fx, fy=fy).astype(np.float32), cv.COLOR_GRAY2BGR)

            if mask is not None:
                mview = cv.cvtColor(cv.resize(mview, (0, 0), fx=fx, fy=fy, interpolation=cv.INTER_NEAREST).astype(np.float32),
                                    cv.COLOR_GRAY2BGR)

                if mview.sum() > 0 and border_only:
                    sparse = coo_matrix(mview[:, :, 0])

                    if multi_labels is not None:
                        for row, col, data in zip(sparse.row, sparse.col, sparse.data):
                            index = int(data*(multi_labels - 1))

                            if multi_labels == HALF_MULTI_TASK_NCHANNELS:
                                keyword = half_multi_task_labels[index]
                            elif multi_labels == MULTI_TASK_NCHANNELS:
                                keyword = multi_task_labels[index].split(' ')[1]
                                if keyword == "globus":
                                    keyword = "globus pallidus"

                            mview[row, col] = np.array(color_map[keyword])

                        for i in range(1, HALF_MULTI_TASK_NCHANNELS):
                            label = half_multi_task_labels[i]
                            imagePrint(mview, label, color=color_map[label], org=(20, 20 + i*20), scale=0.8)
                    else:
                        for row, col in zip(sparse.row, sparse.col):
                            keyword = "hipocampus"
                            mview[row, col] = np.array(color_map[keyword])
                        imagePrint(mview, keyword, color=color_map[keyword], org=(20, 20), scale=0.8)
                else:
                    # Green mask
                    mview[:, :, 0] = 0
                    mview[:, :, 2] = 0

            if ref is not None:
                rview = cv.cvtColor(cv.resize(rview, (0, 0), fx=fx, fy=fy).astype(np.float32), cv.COLOR_GRAY2BGR)

                # Red reference (Bgr)
                rview[:, :, 0] = 0
                rview[:, :, 1] = 0

                mview = mview + rview
                mview[mview > 1] = 1

                sum_view = view + mview
                sum_view[sum_view > 1] = 1

            else:
                if mask is None:
                    mview = None
                else:
                    sum_view = view + mview
                    sum_view[sum_view > 1] = 1

            if mview is not None and sum_view is not None:
                if imprint:
                    imagePrint(view, orientations[j] + " vol " + str(wait) + "ms", scale=0.8)
                    imagePrint(mview, orientations[j] + " mask ", scale=0.8)
                    imagePrint(sum_view, orientations[j] + " overlap ", scale=0.8)
                final_view = np.hstack([view, mview, sum_view])
            else:
                final_view = view

            final_view = imagePrint(final_view, "label: " + str(label), color=(0, 1, 0))
            final_view = imagePrint(final_view, "prediction", color=(0, 1, 0), org=(20, 80))
            final_view = imagePrint(final_view, "reference", color=(0, 0, 1), org=(20, 105))
            if print_slice:
                final_view = imagePrint(final_view, "slice: " + str(k), color=(0, 1, 0), org=(20, 50))
            cv.imshow(id, final_view)

            # User interaction
            key = cv.waitKey(wait)
            if key == ESC:
                if quit_on_esc:
                    quit()
                else:
                    return
            elif key == UP or key == NUP:
                if wait != 0:
                    inc = int(wait/10)
                    if inc == 0:
                        inc = 1
                    wait += inc
                else:
                    wait = 1
            elif key == DOWN or key == NDOWN:
                if wait > 0:
                    dec = int(wait/10)
                    if dec == 0:
                        dec = 1
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
            elif key == D:
                saveimg = (view*255).astype(np.uint8)

                cv.imwrite("data/sample.jpg", saveimg)
                if mask is not None:
                    savemask = (mview*255).astype(np.uint8)
                    cv.imwrite("data/mask.jpg", savemask)
                    borders = cv.Canny(savemask, 100, 200)
                    borders = cv.cvtColor(borders, cv.COLOR_GRAY2RGB)
                    borders[:, :, 0] = 0
                    bgrimg = cv.cvtColor(saveimg, cv.COLOR_GRAY2RGB).astype(np.int)
                    finalsum = bgrimg + borders
                    finalsum[finalsum > 255] = 255
                    finalsum = finalsum.astype(np.uint8)
                    cv.imwrite("data/borders.jpg", finalsum)
                if ref is not None:
                    saveref = (rview*255).astype(np.uint8)
                    cv.imwrite("data/ref.jpg", saveref)

            if kup:
                k += 1
            else:
                k -= 1


def myrotate(inpt, d, order=0):
    '''
    Simplifies rotation calls for 2D images
    '''
    return rotate(inpt, d, resize=True, order=order)


def cprint(msg, out=True):
    '''
    Conditional print
    '''
    if out:
        print(msg)


class Timer():
    def __init__(self, name='', ndecimals=10):
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
        if printout:
            print("Timer {}: {}s".format(self.name, str(end)))
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


def show_multichannel_slice(npinput, npoutput, npref=None, wait=0, multichannel_labels=None):
    for i in range(npoutput.shape[0]):
        for j in range(npoutput.shape[1]):
            inp = npinput[i, 0, :, :] if npinput.shape[1] == 1 else npinput[i, 1, :, :]

            out = npoutput[i, j, :, :]

            if npref is not None:
                ref = npref[i, j, :, :]
            else:
                ref = np.zeros_like(inp)

            mtask = np.hstack((inp, out, ref))

            if multichannel_labels is not None:
                mtask = imagePrint(mtask, "Channel: " + str(j) + ',' + multichannel_labels[j] + " output " + 5*' ' + "target",
                                   scale=0.5)

            cv.imshow("Multitask visualization", mtask)
            if cv.waitKey(wait) == ESC:
                quit()


def imagePrint(img, st, org=(20, 20), scale=1, color=1, font=cv.FONT_HERSHEY_SIMPLEX):
    '''
    Simplifies printing text on images
    '''
    if st is None:
        return img
    else:
        return cv.putText(img, str(st), org, font, scale, color)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


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


# Adapted from genius idea on stackoverflow
def all_idx(idx, axis):
    grid = np.ogrid[tuple(map(slice, idx.shape))]
    grid.insert(axis, idx)
    return tuple(grid)


def ce_output_to_mask(x, numpy=True):
    '''
    Converts cross entropy activations into binary masks for each channel
    '''
    npoutput = np.zeros(x.shape)
    for batch in range(x.shape[0]):
        npoutput[batch] = int_to_onehot(x[batch].max(dim=0)[1].detach().cpu().numpy(),
                                        overhide_max=x.shape[1])

    assert npoutput.sum(axis=1).max() == 1, "multichannel activations with CE loss wrong"

    if numpy:
        return npoutput
    else:
        return torch.from_numpy(npoutput)


def int_to_onehot(matrix, onehot_type=np.dtype(np.float32), overhide_max=None):
    '''
    Converts a matrix of int values (will try to convert) to one hot vectors
    '''
    if overhide_max is None:
        vec_len = int(matrix.max() + 1)
    else:
        vec_len = overhide_max

    onehot = np.zeros((vec_len,) + matrix.shape, dtype=onehot_type)

    int_matrix = matrix.astype(np.int)

    onehot[all_idx(int_matrix, axis=0)] = 1

    return onehot


def get_device(verbose=True):
    '''
    Gets pytorch current available device
    '''
    print("CUDA device available? ", end='')
    if torch.cuda.is_available() is True:
        device = torch.device("cuda:0")
        print("yes: " + str(device))
        print("GPU: " + str(get_device_name(0)))
        GPU_MEMORY_GBS = torch.cuda.get_device_properties(device).total_memory / 1000000000
        print("TOTAL GPU MEMORY: {}".format(GPU_MEMORY_GBS))
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
    db_name = "mnihip"
    notest = False
    study_ths = False
    batch_size = None
    lr = None
    multifigures = False
    dod = False
    rot = False
    wait = 1

    valid_dbs = ["mnihip", "adni", "mniadni", "cc359", "concat", "harp", "multitask", "oldharp", "nathip"]

    if len(argv) > 1:
        if "rot" in argv:
            print("WARNING: Applying random rotations in volumetric test")
            rot = True
        if "dod" in argv:
            print("WARNING: Performing orientation detection insteado of assuming orientation")
            dod = True
        if "multifigures" in argv:
            print("Ploting multifigures")
            multifigures = True
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
        if "batch_size" in argv:
            batch_size = int(argv[argv.index("batch_size") + 1])
            print("Changing batch size to {}".format(batch_size))
        if "lr" in argv:
            lr = float(argv[argv.index("lr") + 1])
            print("Changing LR to {}".format(lr))
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

    return {"display_volume": display_volume, "train": train, "volume": volume, "db_name": db_name, "notest": notest,
            "study_ths": study_ths, "wait": wait, "finetune": finetune, "batch_size": batch_size, "lr": lr,
            "multifigures": multifigures, "nowait": "nowait" in argv, "rot": rot, "dod": dod}


def iarot90(img, axis='X'):
    '''
    Adapted from ia636 toolbox
    '''
    PIVAL = math.pi
    g = 0

    if axis == 'X':
        Trx = np.array([[np.cos(PIVAL/2), -np.sin(PIVAL/2), 0, img.shape[1] - 1],
                        [np.sin(PIVAL/2), np.cos(PIVAL/2),  0, 0],
                        [0,               0,                1, 0],
                        [0,               0,                0, 1]]).astype(float)
        g = iaffine(img, Trx, [img.shape[1], img.shape[0], img.shape[2]])

    elif axis == 'Y':
        Try = np.array([[np.cos(PIVAL/2),  0, np.sin(PIVAL/2), 0],
                        [0,                1, 0,               0],
                        [-np.sin(PIVAL/2), 0, np.cos(PIVAL/2), img.shape[0] - 1],
                        [0,                0, 0,               1]])
        g = iaffine(img, Try, [img.shape[2], img.shape[1], img.shape[0]])
    elif axis == 'Z':
        Trz = np.array([[1, 0,               0,                0],
                        [0, np.cos(PIVAL/2), -np.sin(PIVAL/2), img.shape[2] - 1],
                        [0, np.sin(PIVAL/2), np.cos(PIVAL/2),  0],
                        [0, 0,               0,                1]]).astype(float)
        g = iaffine(img, Trz, [img.shape[0], img.shape[2], img.shape[1]])

    return g


def iaffine(f, T, domain=0):

    if np.sum(domain) == 0:
        domain = f.shape
    if len(f.shape) == 2:
        H, W = f.shape
        y1, x1 = np.indices((domain))

        yx1 = np.array([y1.ravel(),
                        x1.ravel(),
                        np.ones(np.product(domain))])
        yx_float = np.dot(np.linalg.inv(T), yx1)
        yy = np.rint(yx_float[0]).astype(int)
        xx = np.rint(yx_float[1]).astype(int)

        y = np.clip(yy, 0, H - 1).reshape(domain)
        x = np.clip(xx, 0, W - 1).reshape(domain)

        g = f[y, x]

    if len(f.shape) == 3:
        D, H, W = f.shape
        z1, y1, x1 = np.indices((domain))
        zyx1 = np.array([z1.ravel(),
                         y1.ravel(),
                         x1.ravel(),
                         np.ones(np.product(domain))])
        zyx_float = np.dot(np.linalg.inv(T), zyx1)

        zz = np.rint(zyx_float[0]).astype(int)
        yy = np.rint(zyx_float[1]).astype(int)
        xx = np.rint(zyx_float[2]).astype(int)

        z = np.clip(zz, 0, D-1).reshape(domain)  # z
        y = np.clip(yy, 0, H-1).reshape(domain)  # rows
        x = np.clip(xx, 0, W-1).reshape(domain)  # columns

        g = f[z, y, x]

    return g
