'''
Contains custom transform functions supporting a image and a mask as io
To ensure compatibility input should be always numpy

Author: Diedre Carmo
https://github.com/dscarmo
'''
import sys
import time
import os
import torch
import subprocess
import torch.nn.functional as F
from math import inf
import numpy as np
import cv2 as cv
import nibabel as nib
import random
from matplotlib import pyplot as plt
import datetime
from tqdm import tqdm
import sparse as sparse3d
from scipy.sparse import dok_matrix
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import rotate
from utils import myrotate, normalizeMri, get_slice, get_device, error_dialog, class2one_hot, one_hot2dist
from label import get_largest_components
from scipy.ndimage import rotate as rotate3d

MNI_BUFFER_VOL_PATH = os.path.normpath('cache/mnibuffer.nii.gz')
MNI_BUFFER_MASK_PATH = os.path.normpath('cache/mnimaskbuffer.nii.gz')
MNI_BUFFER_MATRIX_PATH = os.path.normpath('cache/mnibuffer.mat')


class Compose(object):
    '''
    Executes all transforms given in tranlist (as a list)
    '''
    def __init__(self, tranlist, time_trans=False):
        self.tranlist = tranlist
        self.time_trans = time_trans

    def addto(self, tran, begin=False, end=False):
        assert begin != end, "either add to begin or end"
        if begin:
            self.tranlist = [tran] + self.tranlist
        elif end:
            self.tranlist = self.tranlist + [tran]

    def __call__(self, img, mask):
        for tran in self.tranlist:
            if self.time_trans:
                begin = time.time()

            img, mask = tran(img, mask)

            if self.time_trans:
                print("{} took {}s".format(tran, time.time() - begin))

        if self.time_trans:
            print("-------- Composed Transforms Finished ---------")

        return img, mask

    def __str__(self):
        string = ""
        for tran in self.tranlist:
            string += str(tran) + ' '
        return string[:-1]


class SoftTarget(object):
    '''
    Randomly transforms hard target in a soft target, according to std parameter
    More STD = softer curve in borders on gaussian mode
    More Order = harder curve in sigmoid mode
    Gaussian still present as backwards compatibility, should not be used for best performance (i hope)
    '''
    def __init__(self, order=10, clip_limit=10, p=0.2, border_focus=False, gaussian=False, dist_type=cv.DIST_L1):

        self.std = order
        self.p = p
        self.border_focus = border_focus
        self.gaussian = gaussian
        self.order = order/10
        self.clip_limit = clip_limit
        self.dist_type = dist_type

    def __call__(self, img, tgt):
        random.seed()
        if random.random() < self.p:
            if self.gaussian:
                soft_tgt = gaussian_filter(tgt, self.std)
                if self.border_focus:
                    inverted_tgt = 1 - tgt
                    inverted_soft_tgt = 1 - soft_tgt
                    outisde_gradient = inverted_tgt*soft_tgt
                    inside_gradient = (inverted_soft_tgt + 0.5)*tgt
                    return img, outisde_gradient + inside_gradient
                else:
                    return img, soft_tgt
            else:
                inverted_tgt = 1 - tgt
                distance_from_original = cv.distanceTransform((inverted_tgt*255).astype(np.uint8), self.dist_type, 3)
                distance_from_inverted = cv.distanceTransform((tgt*255).astype(np.uint8), self.dist_type, 3)
                dist_map = distance_from_inverted - distance_from_original
                clip_dist_map = np.clip(dist_map, -self.clip_limit, self.clip_limit)
                sig_target = torch.from_numpy(clip_dist_map*self.order).sigmoid().numpy()
                return img, sig_target
        else:
            return img, tgt

    def __str__(self):
        return ("SoftTarget with std {} p {} border_focus {}, order {}, gaussian {}, dist {},"
                " clip_limit {}".format(self.std, self.p, self.border_focus, self.order, self.gaussian, self.dist_type,
                                        self.clip_limit))


class CenterCrop(object):
    '''
    Center crops sample and image (should be ndarrays)
    Its never called on patches
    '''
    def __init__(self, cropx, cropy, cropz=None):
        '''
        cropx: crop width
        cropy: crop height
        cropz: crop depth, if not None will consider input a 3D numpy volume
        '''
        self.cropx = cropx
        self.cropy = cropy
        self.cropz = cropz
        if cropz is not None:
            self.volumetric = True
        else:
            self.volumetric = False

    def __call__(self, img, mask):
        '''
        img: 2D numpy array
        mask: 2D numpy array
        '''
        cropx = self.cropx
        cropy = self.cropy
        cropz = self.cropz

        if self.volumetric:
            if img.ndim > 3:
                c, z, y, x = img.shape
            else:
                z, y, x = img.shape
        else:
            if img.ndim > 2:
                c, y, x = img.shape
            else:
                y, x = img.shape

        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)
        if self.volumetric:
            startz = z//2-(cropz//2)
            if img.ndim > 3:
                rimg = img[:, startz:startz+cropz, starty:starty+cropy, startx:startx+cropx]
            else:
                rimg = img[startz:startz+cropz, starty:starty+cropy, startx:startx+cropx]

            if mask.ndim > 3:
                rmask = mask[:, startz:startz+cropz, starty:starty + cropy, startx:startx+cropx]
            else:
                rmask = mask[startz:startz+cropz, starty:starty + cropy, startx:startx+cropx]

            return rimg, rmask
        else:  # considers multi channels slices (ndim > 2)
            if img.ndim > 2:
                ret_img = img[:, starty:starty + cropy, startx:startx+cropx]
            else:
                ret_img = img[starty:starty + cropy, startx:startx + cropx]

            if mask.ndim > 2:
                ret_mask = mask[:, starty:starty + cropy, startx:startx+cropx]
            else:
                ret_mask = mask[starty:starty + cropy, startx:startx + cropx]

            return ret_img, ret_mask

    def __str__(self):
        return "CenterCrop, patch size: {}x{}x{} volumetric {}".format(self.cropx, self.cropy, self.cropz, self.volumetric)


class ToTensor(object):
    '''
    Convert ndarrays in sample to Tensors.
    '''
    def __init__(self, volumetric=False, rune2d=False, e2d_device=None, pre_saved_model=None, small=False,
                 transform_to_onehot=False, C=None, return_dists=False, addch=False, debug=False):
        '''
        volumetric: indicates if the input is 3D
        rune2d: indicates if e2d consensus should be pre computed (used in E3D models)
        e2d_device: device used for e2d computation (can be different than device used in main flow)
        pre_saved_model: folder with saved weights of pre saved e2d model
        small: wether using a smaller model or not
        '''
        self.debug = debug
        self.volumetric = volumetric
        if rune2d:
            assert self.volumetric and pre_saved_model is not None, ("to rune2d in ToTensor, you need volumetric input and a "
                                                                     "pre_saved_model!")
        self.pre_saved_model = pre_saved_model
        self.rune2d = rune2d
        if e2d_device is None and self.rune2d:
            self.e2d_device = get_device(verbose=False)
        else:
            self.e2d_device = e2d_device
        self.small = small
        self.transform_to_onehot = transform_to_onehot
        self.C = C
        self.return_dists = return_dists
        self.addch = addch

    def __call__(self, npimage, npmask):
        '''
        input numpy image: H x W
        output torch image: C X H X W
        '''
        if self.debug:
            ttinput = npimage[1] + npmask
            ttinput[ttinput > 1.0] = 1.0
            ttinput[ttinput < 0] = 0
            plt.figure(num="ToTensor input")
            plt.imshow(ttinput, cmap='gray', vmax=1.0, vmin=0.0)
            plt.figure(num="ToTensor input image")
            plt.imshow(npimage[1], cmap='gray', vmax=1.0, vmin=0.0)
            plt.figure(num="ToTensor input mask")
            plt.imshow(npmask, cmap='gray', vmax=1.0, vmin=0.0)
        if self.rune2d:
            nchans = 2

            ishape = (nchans, npimage.shape[0], npimage.shape[1], npimage.shape[2])
            image = torch.zeros(ishape).float()

            np_input, np_e2d_output = run_once(None, self.pre_saved_model, numpy_input=npimage, save=False, verbose=False,
                                               device=self.e2d_device, small=self.small, filter_components=False,
                                               addch=self.addch)

            image[0] = torch.from_numpy(np_input).float()
            image[1] = torch.from_numpy(np_e2d_output).float()

        elif npimage.ndim == 2 or (len(npimage.shape) == 3 and self.volumetric):
            image = torch.unsqueeze(torch.from_numpy(npimage), 0).float()
        else:
            image = torch.from_numpy(npimage).float()

        if npmask.ndim == 2:
            mask = torch.unsqueeze(torch.from_numpy(npmask), 0).float()
        else:
            mask = torch.from_numpy(npmask).float()

        image[image > 1.0] = 1.0
        image[image < 0.0] = 0.0
        mask[mask > 1.0] = 1.0
        mask[mask < 0.0] = 0.0

        if self.transform_to_onehot:
            if mask.shape[0] == 1:  # check if mask is in one channel format (should be if there is only 1 class)
                mask = class2one_hot(mask[0], self.C)[0]
                if self.return_dists:
                    mask = (mask, torch.from_numpy(one_hot2dist(mask.float().numpy())))
            else:
                raise ValueError("Dont know what to do with this shape in ToTensor {}".format(mask.shape))

        return image, mask

    def __str__(self):
        return ("ToTensor, volumetric: {}, rune2d: {}, e2d_device: {}, pre_saved_model passed: {}, "
                "small: {}".format(self.volumetric, self.rune2d, self.e2d_device, self.pre_saved_model is not None, self.small))


class ToNumpy(object):
    '''
    Convert tensors in sample to ndarrays.
    '''
    def __call__(self, image, mask=None):
        '''
        input torch image: C X H X W
        output numpy image: H x W
        '''
        npimage = torch.squeeze(image).numpy()
        npimage[npimage > 1.0] = 1.0
        npimage[npimage < 0.0] = 0.0
        if mask is not None:
            npmask = torch.squeeze(mask).numpy()
            npmask[npmask > 1.0] = 1.0
            npmask[npmask < 0.0] = 0.0
            return npimage, npmask
        else:
            return npimage

    def __str__(self):
        return "ToTensor"


class ReturnPatch(object):
    '''
    Random patch centered around hippocampus
    If no hippocampus present, random patch
    Ppositive is chance of returning a patch around the hippocampus
    Kernel shape is shape of kernel for boundary extraction

    In current state, Multitask selects a random patch
    '''
    def __init__(self, ppositive=0.8, patch_size=(32, 32), kernel_shape=(3, 3), fullrandom=False, anyborder=False, debug=False):
        '''
        Sets desired patchsize (width, height)
        '''
        self.psize = patch_size
        self.ppositive = ppositive
        self.kernel = np.ones(kernel_shape, np.uint8)
        self.ks = kernel_shape
        self.fullrandom = fullrandom
        self.anyborder = anyborder
        self.debug = debug
        dim = len(patch_size)
        assert dim in (2, 3), "only support 2D or 3D patch"
        if dim == 3:
            self.volumetric = True
        elif dim == 2:
            self.volumetric = False

    def random_choice_3d(self, keylist):
        '''
        Returns random point in 3D sparse COO object
        '''
        lens = [len(keylist[x]) for x in range(3)]
        assert lens[0] == lens[1] and lens[0] == lens[2] and lens[1] == lens[2], "error in random_choice_3d sparse matrix"
        position = random.choice(range(len(keylist[0])))
        point = [keylist[x][position] for x in range(3)]
        return point

    def __call__(self, image, mask, debug=False):
        '''
        Returns patch of image and mask
        '''
        debug = self.debug
        random.seed()
        # Get list of candidates for patch center
        e2d = False
        shape = image.shape
        if not self.volumetric and len(shape) == 3:
            shape = (shape[1], shape[2])
            e2d = True

        if not self.fullrandom:
            if self.volumetric:
                borders = np.zeros(shape, dtype=mask.dtype)
                for i in range(shape[0]):
                    uintmask = (mask[i]*255).astype(np.uint8)
                    borders[i] = ((uintmask - cv.erode(uintmask, self.kernel, iterations=1))/255).astype(mask.dtype)
                sparse = sparse3d.COO.from_numpy(borders)
                keylist = sparse.nonzero()
            else:
                if mask.ndim > 2:
                    # hmask is now everything, hip only deprecated in multitask
                    if self.anyborder:
                        hmask = mask.sum(axis=0)
                    else:
                        try:
                            hmask = mask[11] + mask[12]
                        except IndexError:  # half labels
                            hmask = mask[6]
                else:
                    hmask = mask
                # Get border of mask
                uintmask = (hmask*255).astype(np.uint8)
                borders = ((uintmask - cv.erode(uintmask, self.kernel, iterations=1))/255).astype(hmask.dtype)
                sparse = dok_matrix(borders)
                keylist = list(sparse.keys())
                if debug:
                    print("Candidates {}".format(keylist))

        # Get top left and bottom right of patch centered on mask border
        tl_row_limit = shape[0] - self.psize[0]
        tl_col_limit = shape[1] - self.psize[1]
        if self.volumetric:
            tl_depth_limit = shape[2] - self.psize[2]
            tl_rdepth = inf
        tl_rrow = inf
        tl_rcol = inf

        if self.fullrandom:
            if self.volumetric:
                tl_rrow, tl_rcol, tl_rdepth = (random.randint(0, tl_row_limit), random.randint(0, tl_col_limit),
                                               random.randint(0, tl_depth_limit))
            else:
                tl_rrow, tl_rcol = random.randint(0, tl_row_limit), random.randint(0, tl_col_limit)
        elif len(keylist[0]) > 0 and random.random() < self.ppositive:
            if self.volumetric:
                while tl_rrow > tl_row_limit or tl_rcol > tl_col_limit or tl_rdepth > tl_depth_limit:
                    tl_rrow, tl_rcol, tl_rdepth = self.random_choice_3d(keylist)
                    tl_rrow -= self.psize[0]//2
                    tl_rcol -= self.psize[1]//2
                    tl_rdepth -= self.psize[2]//2
            else:
                while tl_rrow > tl_row_limit or tl_rcol > tl_col_limit:
                    tl_rrow, tl_rcol = random.choice(list(sparse.keys()))
                    tl_rrow -= self.psize[0]//2
                    tl_rcol -= self.psize[1]//2
        else:
            if self.volumetric:
                tl_rrow, tl_rcol, tl_rdepth = (random.randint(0, tl_row_limit), random.randint(0, tl_col_limit),
                                               random.randint(0, tl_depth_limit))
            else:
                tl_rrow, tl_rcol = random.randint(0, tl_row_limit), random.randint(0, tl_col_limit)

        if tl_rrow < 0:
            tl_rrow = 0
        if tl_rcol < 0:
            tl_rcol = 0
        if self.volumetric:
            if tl_rdepth < 0:
                tl_rdepth = 0

        if debug:
            print("Patch top left(row, col): {} {}".format(tl_rrow, tl_rcol))

        if self.volumetric:
            rimage = image[tl_rrow:tl_rrow + self.psize[0], tl_rcol:tl_rcol + self.psize[1], tl_rdepth:tl_rdepth + self.psize[2]]
            rmask = mask[tl_rrow:tl_rrow + self.psize[0], tl_rcol:tl_rcol + self.psize[1], tl_rdepth:tl_rdepth + self.psize[2]]
            assert rimage.shape == self.psize and rmask.shape == self.psize, ("fatal error generating patches, incorrect patch"
                                                                              "size image: {}, mask: {}, intended patch size:"
                                                                              "{}".format(rimage.shape, rmask.shape, self.psize))
        else:
            if e2d:
                rimage = image[:, tl_rrow:tl_rrow + self.psize[0], tl_rcol:tl_rcol + self.psize[1]]
            else:
                rimage = image[tl_rrow:tl_rrow + self.psize[0], tl_rcol:tl_rcol + self.psize[1]]

            if len(mask.shape) > 2:
                rmask = mask[:, tl_rrow:tl_rrow + self.psize[0], tl_rcol:tl_rcol + self.psize[1]]
            else:
                rmask = mask[tl_rrow:tl_rrow + self.psize[0], tl_rcol:tl_rcol + self.psize[1]]

        if debug:
            print(rimage.shape, rmask.shape)
            from matplotlib import pyplot as plt
            fulldisp = image[1] + mask
            fulldisp[fulldisp > 1] = 1
            fulldisp[fulldisp < 0] = 0
            disp = rimage[1] + rmask
            disp[disp > 1] = 1
            disp[disp < 0] = 0
            plt.figure(num="overlap")
            plt.imshow(fulldisp, cmap='gray', vmin=0, vmax=1)
            plt.figure(num="Patch overlap")
            plt.imshow(disp, cmap='gray', vmin=0, vmax=1)
            plt.figure(num="Mask Patch")
            plt.imshow(rmask, cmap='gray', vmin=0, vmax=1)
            plt.figure(num="Brain Patch")
            plt.imshow(rimage[1], cmap='gray', vmin=0, vmax=1)
            plt.figure(num="Borders")
            plt.imshow(borders, cmap='gray', vmin=0, vmax=1)
            plt.figure(num="Brain")
            plt.imshow(image[1], cmap='gray', vmin=0, vmax=1)
            plt.figure(num="Mask")
            plt.imshow(mask, cmap='gray', vmin=0, vmax=1)

        return rimage, rmask

    def __str__(self):
        return ("ReturnPatch: ppositive {} patch_size {}, kernel_shape {}, volumetric {}, "
                "anyborder {}".format(self.ppositive, self.psize, self.ks, self.volumetric, self.anyborder))


class ToFloat32(object):
    '''
    Transforms stored float16 to float32
    '''
    def __call__(self, image, mask):
        return image.astype(np.float32), mask.astype(np.float32)

    def __str__(self):
        return "ToFloat32"


class RandomFlip(object):
    '''
    Randomly flip vertically, or horizontally
    1-p % chance to not do anything
    if doing something, sort between modes=( horflip, verflip, both)
    change modes variable to remove any of these or chance
    Doesnt make much sense in the 3D big patch case.
    '''
    def __init__(self, p=0.2, modes=['horflip', 'verflip', 'both']):
        self.p = p
        self.modes = modes

    def __call__(self, image, mask):
        random.seed()
        if random.random() < self.p:
            op = random.choice(self.modes)

            if op == 'horflip':
                if mask.ndim > 2:
                    for j in range(mask.shape[0]):
                        mask[j] = np.fliplr(mask[j]).copy()
                else:
                    mask = np.fliplr(mask).copy()

                if image.ndim > 2:
                    for j in range(image.shape[0]):
                        image[j] = np.fliplr(image[j]).copy()
                else:
                    image = np.fliplr(image).copy()
            elif op == 'verflip':
                if mask.ndim > 2:
                    for j in range(mask.shape[0]):
                        mask[j] = np.flipud(mask[j]).copy()
                else:
                    mask = np.flipud(mask).copy()

                if image.ndim > 2:
                    for j in range(image.shape[0]):
                        image[j] = np.flipud(image[j]).copy()
                else:
                    image = np.flipud(image).copy()
            elif op == 'both':
                if mask.ndim > 2:
                    for j in range(mask.shape[0]):
                        mask[j] = np.fliplr(np.flipud(mask[j])).copy()
                else:
                    mask = np.fliplr(np.flipud(mask)).copy()

                if image.ndim > 2:
                    for j in range(image.shape[0]):
                        image[j] = np.fliplr(np.flipud(image[j])).copy()
                else:
                    image = np.fliplr(np.flipud(image)).copy()
            else:
                raise ValueError("Invalid RandomFlip mode: {}".format(op))

        return image, mask

    def __str__(self):
        return "RandomFlip: p {}, modes {}".format(self.p, self.modes)


class Intensity(object):
    '''
    Randomly applies intensity transform to the image
    Image should be between 0 and 1
    '''
    def __init__(self, p=1, brightness=0.10, force_max=False):
        assert p > 0 and p <= 1 and brightness > 0 and brightness <= 1, "arguments make no sense"
        self.p = p
        self.b = brightness
        self.force_max = force_max

    def __call__(self, image, mask):
        random.seed()
        if random.random() < self.p:
            if self.force_max:
                value = self.b
            else:
                value = ((random.random() - 0.5)*2)*self.b
            image += value

        return image, mask

    def __str__(self):
        return "Intensity: p {}, brightness {}".format(self.p, self.b)


class Noisify(object):
    '''
    Randomly adds gaussian or poison noise
    '''
    def __init__(self, noise="gaussian", p=0.2, var=0.0002, mean=0):
        self.p = p
        self.var = var
        self.mean = mean
        self.sigma = var**0.5
        self.noise = noise
        supported_noises = ["gaussian", "poisson"]
        assert noise in supported_noises, "unsupported noise {} in Noisify, should be {}".format(noise, supported_noises)

    def __call__(self, image, mask):
        random.seed()
        if random.random() < self.p:
            if self.noise == "gaussian":
                shape = image.shape
                gauss = np.random.normal(self.mean, self.sigma, shape)
                gauss = gauss.reshape(*shape)
                inoisy = image + gauss
            elif self.noise == "poisson":
                noise_mask = np.random.poisson(image)
                inoisy = image + noise_mask
            return inoisy, mask
        else:
            return image, mask

    def __str__(self):
        return "Noisify: noise {}, p {}, var {}, mean {}, sigma {}".format(self.noise, self.p, self.var, self.mean, self.sigma)


class RandomAffine(object):
    '''
    Randomly applies scale -> rotation to simulate different data source
    WARNING: Input has to have even shape, will be fixed soon
    TODO: Volumetric support
    '''
    LESS = -1
    EQUAL = 0
    MORE = 1

    def __init__(self, p=0.2, rotate=20, scale=(0.8, 1.2), fillmode='constant', volumetric=False, debug=False):
        self.p = p
        self.rotate = rotate
        self.scale = scale
        self.fmode = fillmode
        self.volumetric = volumetric
        self.debug = debug

    def __call__(self, image, mask):
        debug = self.debug
        random.seed()
        if random.random() < self.p:
            # Save original size for final center crop
            original_size = np.array(image.shape)

            im_ndim = image.ndim
            ma_ndim = mask.ndim

            if im_ndim > 2:
                original_size = np.array((original_size[1], original_size[2]))

            self.center_croper = CenterCrop(original_size[1], original_size[0])
            if debug:
                print("im {} mask {}".format(image.shape, mask.shape))
                print("Original size {}".format(original_size))
                print("IMDIM {}".format(im_ndim))
                print("MASKDIM {}".format(ma_ndim))
            # Pick a scale factor
            scale = random.uniform(self.scale[0], self.scale[1])

            # Target scale size
            scale_size = (scale*original_size).astype(np.int)

            # If odd, add + 1 for simplicity of padding later
            if scale_size[0] % 2:
                scale_size += 1

            # Apply scale
            self.resizer = Resize(scale_size[1], scale_size[0])
            if debug:
                print("pre resize mask {}")

            image, mask = self.resizer(image, mask)

            if debug:
                print("post resize mask {}")
                print("After resize im{}".format(image.shape))

            # Apply rotation
            ang = 2*(random.random() - 0.5)*self.rotate
            if debug:
                print("Rotating by {}".format(ang))
            if im_ndim > 2:
                for i in range(image.shape[0]):
                    image[i] = rotate(image[i], ang, preserve_range=True, mode=self.fmode, order=3)
            else:
                image = rotate(image, ang, preserve_range=True, mode=self.fmode, order=3)

            if ma_ndim > 2:
                for i in range(mask.shape[0]):
                    mask[i] = rotate(mask[i], ang, preserve_range=True, mode=self.fmode, order=0)
            else:
                mask = rotate(mask, ang, preserve_range=True, mode=self.fmode, order=0)

            if debug:
                print("After rot shape {}".format(image.shape))

            # Go back to original size
            shape_diff, (rowdiff, coldiff) = self.compare_shapes(image.shape, original_size)
            if shape_diff == self.LESS:
                if im_ndim > 2:
                    buffer = np.zeros((image.shape[0], original_size[0], original_size[1]), dtype=image.dtype)
                    for i in range(image.shape[0]):
                        buffer[i] = np.pad(image[i], (rowdiff//2, coldiff//2), mode=self.fmode)
                    image = buffer
                else:
                    image = np.pad(image, (rowdiff//2, coldiff//2), mode=self.fmode)

                if ma_ndim > 2:
                    buffer = np.zeros((mask.shape[0], original_size[0], original_size[1]), dtype=mask.dtype)
                    for i in range(mask.shape[0]):
                        buffer[i] = np.pad(mask[i], (rowdiff//2, coldiff//2), mode=self.fmode)
                    mask = buffer
                else:
                    if debug:
                        print("padding mask lul {} {}".format(rowdiff, coldiff))
                    mask = np.pad(mask, (rowdiff//2, coldiff//2), mode=self.fmode)

            elif shape_diff == self.EQUAL:
                pass
            elif shape_diff == self.MORE:
                image, mask = self.center_croper(image, mask)

            '''if e2d:
                cv.imshow("image", image[1])
            else:
                cv.imshow("image", image)
            cv.imshow("mask", mask)
            if cv.waitKey(0) == 27:
                quit()'''

            if debug:
                print("Final shapes: {}, {}".format(image.shape, mask.shape))

        return image, mask

    def __str__(self):
        return "RandomAffine: p {}, rotate {}, scale {}, fmode {}, volumetric {}".format(self.p, self.rotate, self.scale,
                                                                                         self.fmode, self.volumetric)

    def compare_shapes(self, shape1, refshape):
        '''
        shape1 > refshape = 1
        shape1 == refshape = 0
        shape1 < refshape = -1
        '''
        add = 0
        if len(shape1) == 3:
            # Add 1 to first shape if E2D
            add = 1

        rowdiff = refshape[0] - shape1[0 + add]
        coldiff = refshape[1] - shape1[1 + add]

        if shape1[0 + add] > refshape[0] and shape1[1 + add] > refshape[1]:
            return self.MORE, (rowdiff, coldiff)
        elif shape1[0 + add] == refshape[0] and shape1[1 + add] == refshape[1]:
            return self.EQUAL, (rowdiff, coldiff)
        elif shape1[0 + add] < refshape[0] and shape1[1 + add] < refshape[1]:
            return self.LESS, (rowdiff, coldiff)
        else:
            raise RuntimeError("Difference in shapes in RandomAffine not supported {} {}".format(shape1, refshape))


class Resize(object):
    '''
    Resize sample and mask, works only in 2D slices
    '''
    def __init__(self, width, height, volumetric=False):
        '''
        size: (width, height)
        '''
        self.size = (width, height)
        self.volumetric = volumetric  # TODO

    def __call__(self, image, mask):
        '''
        Image and mask should be numpy 2D arrays or E2D
        '''
        if image.ndim > 2:
            nimage = np.zeros((image.shape[0], self.size[1], self.size[0]), dtype=image.dtype)
            for i in range(image.shape[0]):
                nimage[i] = cv.resize(image[i], self.size, interpolation=cv.INTER_CUBIC)
        else:
            nimage = cv.resize(image, self.size, interpolation=cv.INTER_CUBIC)

        if mask.ndim > 2:
            nmask = np.zeros((mask.shape[0], self.size[1], self.size[0]), dtype=mask.dtype)
            for i in range(mask.shape[0]):
                nmask[i] = cv.resize(mask[i], self.size, interpolation=cv.INTER_NEAREST)
        else:
            nmask = cv.resize(mask, self.size, interpolation=cv.INTER_NEAREST)

        return nimage, nmask

    def __str__(self):
        return "Resize: size {}, volumetric {}".format(self.size, self.volumetric)


class REGWorker():
    '''
    Avoid multithread conflicts by appending a ID to the beginning of intermediary cache files
    '''
    def __init__(self, worker_id):
        self.worker_id = worker_id

    def add_worker_id(self, inpath):
        return os.path.join(os.path.dirname(inpath), self.worker_id + os.path.basename(inpath))


class MNITransform(object):
    '''
    Input has to be 3D numpy volumes
    '''
    def __call__(self, invol, mask):
        assert len(invol.shape) == 3 and len(mask.shape) == 3, "mnitransform only works with 3D volumes"
        worker_id = datetime.datetime.now().isoformat()
        vol_cache_path = os.path.join("cache", worker_id + "vol.nii.gz")
        mask_cache_path = os.path.join("cache", worker_id + "mask.nii.gz")
        nib.save(nib.nifti1.Nifti1Image(invol, affine=None), vol_cache_path)
        nib.save(nib.nifti1.Nifti1Image(mask, affine=None), mask_cache_path)
        ret = mni152reg(vol_cache_path, mask=mask_cache_path, worker_id=worker_id)
        try:
            os.remove(vol_cache_path)
            os.remove(mask_cache_path)
        except OSError as oe:
            print("WARNING! Problem removing MNIReg path: {}".format(oe))
        return ret

    def __str__(self):
        return "MNITransform"


def perform_random_rotation(x, tgt=None):
    '''
    Performs a random -90 or 90 degrees in one of the three orthogonal axis
    '''
    rot_choice = random.randint(0, 7)

    if rot_choice == 0:  # original mirrored (sagital, coronal, axial)
        x = x[:, ::-1, :].copy()
        if tgt is None:
            tgt = torch.Tensor([0]).long()
        else:
            tgt = tgt[:, ::-1, :].copy()
    elif rot_choice == 1:  # original orientation (sagital, coronal, axial)
        if tgt is None:
            tgt = torch.Tensor([0]).long()
    elif rot_choice == 2:
        x = rotate3d(x, 90, axes=(0, 1), order=0, reshape=True)  # ["coronal", "sagital", "axial"]
        if tgt is None:
            tgt = torch.Tensor([1]).long()
        else:
            tgt = rotate3d(tgt, 90, axes=(0, 1), order=0, reshape=True)  # ["coronal", "sagital", "axial"]
    elif rot_choice == 3:
        x = rotate3d(x, -90, axes=(0, 1), order=0, reshape=True)  # ["coronal", "sagital", "axial"]
        if tgt is None:
            tgt = torch.Tensor([1]).long()
        else:
            tgt = rotate3d(tgt, -90, axes=(0, 1), order=0, reshape=True)  # ["coronal", "sagital", "axial"]
    elif rot_choice == 4:
        x = rotate3d(x, 90, axes=(0, 2), order=0, reshape=True)  # ["axial", "coronal", "sagital"]
        if tgt is None:
            tgt = torch.Tensor([2]).long()
        else:
            tgt = rotate3d(tgt, 90, axes=(0, 2), order=0, reshape=True)  # ["axial", "coronal", "sagital"]
    elif rot_choice == 5:
        x = rotate3d(x, -90, axes=(0, 2), order=0, reshape=True)  # ["axial", "coronal", "sagital"]
        if tgt is None:
            tgt = torch.Tensor([2]).long()
        else:
            tgt = rotate3d(tgt, -90, axes=(0, 2), order=0, reshape=True)  # ["axial", "coronal", "sagital"]
    elif rot_choice == 6:
        x = rotate3d(x, 90, axes=(1, 2), order=0, reshape=True)  # ["sagital", "axial", "coronal"]
        if tgt is None:
            tgt = torch.Tensor([3]).long()
        else:
            tgt = rotate3d(tgt, 90, axes=(1, 2), order=0, reshape=True)  # ["sagital", "axial", "coronal"]
    elif rot_choice == 7:
        x = rotate3d(x, -90, axes=(1, 2), order=0, reshape=True)  # ["sagital", "axial", "coronal"]
        if tgt is None:
            tgt = torch.Tensor([3]).long()
        else:
            tgt = rotate3d(tgt, -90, axes=(1, 2), order=0, reshape=True)  # ["sagital", "axial", "coronal"]

    return x, tgt


def mni152reg(invol, mask=None, ref_brain="/usr/local/fsl/data/standard/MNI152lin_T1_1mm.nii.gz", save_path=MNI_BUFFER_VOL_PATH,
              mask_save_path=MNI_BUFFER_MASK_PATH, remove=True, return_numpy=True, keep_matrix=False, worker_id=''):
    '''
    Register a sample and (optionally) a mask from disk and returns them as numpy volumes
    '''
    reg_worker = REGWorker(worker_id)

    save_path = reg_worker.add_worker_id(save_path)
    mask_save_path = reg_worker.add_worker_id(mask_save_path)

    matrix_buffer = MNI_BUFFER_MATRIX_PATH
    matrix_buffer = reg_worker.add_worker_id(matrix_buffer)

    my_env = os.environ.copy(); my_env["FSLOUTPUTTYPE"] = "NIFTI_GZ" # set FSLOUTPUTTYPE=NIFTI_GZ
    if not os.path.isfile(ref_brain): #if FSL template not found use local copy
        if sys.platform == "win32":
           try: ref_brain = sys._MEIPASS+'\\templates\\MNI152lin_T1_1mm.nii.gz' # when running frozen with pyInstaller
           except: ref_brain = 'templates\\MNI152lin_T1_1mm.nii.gz'            # when running normally 
        else: ref_brain = 'templates/MNI152lin_T1_1mm.nii.gz'            
    ref_brain = os.path.normpath(ref_brain) # use OS specific filename    
    
    if sys.platform == "win32":
        try: flirt_executable = sys._MEIPASS+'\\flirt.exe' # when running frozen with pyInstaller
        except: flirt_executable = 'flirt.exe'            # when running normally 
    else: flirt_executable = 'flirt'    
    
    try:
        ret = None
        subprocess.run([flirt_executable, "-in",  invol, "-ref", ref_brain, "-out", save_path, "-omat", matrix_buffer], env=my_env)
        if return_numpy:
            vol = nib.load(save_path).get_fdata()

        if mask is None and return_numpy:
            ret = vol
        else:
            subprocess.run([flirt_executable, "-in",  mask, "-ref", ref_brain, "-out", mask_save_path, "-init", matrix_buffer,
                            "-applyxfm"], env=my_env)
            if return_numpy:
                mask = nib.load(mask_save_path).get_fdata()
                ret = (vol, mask)

        if remove:
            try:
                os.remove(save_path)
                if not keep_matrix:
                    os.remove(matrix_buffer)
                if mask is not None:
                    os.remove(mask_save_path)
            except OSError as oe:
                print("Error trying to remove mni register buffer files: {}".format(oe))
    except FileNotFoundError as fnfe:
        error_dialog("FLIRT registration error or FLIRT installation not found. Make sure FLIRT is installed for your OS.")
        print("Registration ERROR: {}".format(fnfe))
        sys.exit(1)

    return ret


def run_once(volpath, models, numpy_input=None, save=True, verbose=True, device=None, small=False, return_mask_path=False,
             filter_components=True, addch=False):
    '''
    Runs our best model in a provided volume and saves mask,
    In a self contained matter
    To serve a ready (normalized) numpy_input, volpath should be None
    '''
    assert volpath is not None or numpy_input is not None, "volpath or numpy input should not be None"
    begin = time.time()

    if device is None:
        device = get_device()

    if small:
        CROP_SHAPE = 64
    else:
        CROP_SHAPE = 160
    slice_transform = Compose([CenterCrop(CROP_SHAPE, CROP_SHAPE), ToTensor()])

    if numpy_input is not None:
        sample_v = numpy_input
    else:
        sample_v = normalizeMri(nib.load(volpath).get_fdata().astype(np.float32))

    orientations = ["sagital", "coronal", "axial"]
    print("Assuming {} orientations in volume. If your volume is not in this orientation, run MNI152 reg"
          " or you might get wrong results.".format(orientations))

    shape = sample_v.shape
    sum_vol_total = torch.zeros(shape)

    for o, model in models.items():
        model.eval()
        model.to(device)

    if verbose:
        print("Performing segmentation...")

    for i, o in enumerate(orientations):
        if verbose:
            print("Processing {} view...".format(o))

        slice_shape = get_slice(sample_v, 0, o, orientations=orientations).shape

        if verbose:
            itr = tqdm(range(shape[i]))
        else:
            itr = range(shape[i])

        for j in itr:
            # E2D
            ts = np.zeros((3, slice_shape[0], slice_shape[1]), dtype=np.float32)
            for ii, jj in enumerate(range(j-1, j+2)):
                if jj < 0:
                    jj = 0
                elif jj == shape[i]:
                    jj = shape[i] - 1

                if i == 0:
                    ts[ii] = myrotate(sample_v[jj, :, :], 90)
                elif i == 1:
                    ts[ii] = myrotate(sample_v[:, jj, :], 90)
                elif i == 2:
                    ts[ii] = myrotate(sample_v[:, :, jj], 90)

            s, _ = slice_transform(ts, ts[1])  # work around, no mask
            s = s.to(device)

            probs = models[o](s.unsqueeze(0))

            if addch:
                probs = probs[:, 1, :, :]  # get hip only

            cpup = probs.squeeze().detach().cpu()
            finalp = torch.from_numpy(myrotate(cpup.numpy(), -90)).float()  # back to volume orientation

            # Add to final consensus volume, uses original orientation/shape
            if i == 0:
                toppad = shape[1]//2 - CROP_SHAPE//2
                sidepad = shape[2]//2 - CROP_SHAPE//2

                tf = 1 if shape[1] % 2 == 1 else 0
                sf = 1 if shape[2] % 2 == 1 else 0
                pad = F.pad(finalp, (sidepad + sf, sidepad, toppad, toppad + tf))/3

                sum_vol_total[j, :, :] += pad
            elif i == 1:
                toppad = shape[0]//2 - CROP_SHAPE//2
                sidepad = shape[2]//2 - CROP_SHAPE//2

                tf = 1 if shape[0] % 2 == 1 else 0
                sf = 1 if shape[2] % 2 == 1 else 0
                pad = F.pad(finalp, (sidepad + sf, sidepad, toppad, toppad + tf))/3

                sum_vol_total[:, j, :] += pad
            elif i == 2:
                toppad = shape[0]//2 - CROP_SHAPE//2
                sidepad = shape[1]//2 - CROP_SHAPE//2

                tf = 1 if shape[0] % 2 == 1 else 0
                sf = 1 if shape[1] % 2 == 1 else 0
                pad = F.pad(finalp, (sidepad + sf, sidepad, toppad, toppad + tf))/3

                sum_vol_total[:, :, j] += pad

    if filter_components:
        if verbose:
            print("Performing consensus...")
        final_nppred = get_largest_components(sum_vol_total.numpy(), mask_ths=0.5)
        if verbose:
            print("Consensus done.")
    else:
        if verbose:
            print("Skipping consensus...")
        final_nppred = sum_vol_total.numpy()

    if save:
        if volpath is not None:
            save_path = volpath + "_voxelcount-{}_e2dhipmask.nii.gz".format(int(final_nppred.sum()))
        else:
            save_path = datetime.datetime.now().isoformat() + "_voxelcount-{}_e2dhipmask.nii.gz".format(int(final_nppred.sum()))

        nib.save(nib.nifti1.Nifti1Image(final_nppred, None), save_path)

    if verbose:
        print("Finished segmentation in {}s".format(time.time() - begin))

    if return_mask_path:
        return sample_v, final_nppred, save_path
    else:
        return sample_v, final_nppred


def get_data_transforms(args, **kwargs):
    # Define transforms
    if not args["volumetric"]:
        if args["patch64"]:
            print("USING 64 PATCH")
            patch_size = 64
        else:
            print("USING 32 PATCH")
            patch_size = 32

        # Aug selecting
        selected = 0
        if args["ROTATIONINVARIANT"]:
            print("ROTATION INVARIANT")
            selected += 1
            train_transforms = Compose([ReturnPatch(patch_size=(patch_size, patch_size), anyborder=args["anyborder"],
                                                    fullrandom=args["FULLRANDOM"]),
                                        RandomFlip(modes=['horflip', 'verflip', 'both']),
                                        RandomAffine(rotate=10, scale=(0.9, 1.1)), Intensity(brightness=0.05), Noisify(),
                                        ToTensor(transform_to_onehot=(args["gdl"] or args["boundary"]) and not args["multitask"],
                                                 C=kwargs["nlabels"], return_dists=args["boundary"])])
        elif args["INTFLIP"]:
            selected += 1
            train_transforms = Compose([ReturnPatch(patch_size=(patch_size, patch_size), anyborder=args["anyborder"],
                                                    fullrandom=args["FULLRANDOM"]), RandomFlip(modes=['horflip']),
                                        RandomAffine(), Intensity(), Noisify(), ToTensor()])
        elif args["INTNOFLIP"]:
            print("NO FLIP IN AUGMENTATION")
            selected += 1
            train_transforms = Compose([ReturnPatch(patch_size=(patch_size, patch_size), anyborder=args["anyborder"],
                                                    fullrandom=args["FULLRANDOM"]),
                                        RandomAffine(), Intensity(), Noisify(), ToTensor()])
        elif args["halfaug"]:
            print("NO FLIP IN AUGMENTATION HALF AUG")
            selected += 1
            train_transforms = Compose([ReturnPatch(patch_size=(patch_size, patch_size), anyborder=args["anyborder"],
                                                    fullrandom=args["FULLRANDOM"]),
                                        RandomAffine(rotate=10, scale=(0.9, 1.1)), Intensity(brightness=0.05), Noisify(),
                                        ToTensor(transform_to_onehot=(args["gdl"] or args["boundary"]) and not args["multitask"],
                                                 C=kwargs["nlabels"], return_dists=args["boundary"])])
        elif args["NOINTNOFLIP"]:  # defacto aug fog harp
            print("NO FLIP IN AUGMENTATION HALF AUG")
            selected += 1
            train_transforms = Compose([ReturnPatch(patch_size=(patch_size, patch_size), anyborder=args["anyborder"],
                                                    fullrandom=args["FULLRANDOM"]),
                                        RandomAffine(rotate=10, scale=(0.9, 1.1)), Noisify(),
                                        ToTensor(transform_to_onehot=(args["gdl"] or args["boundary"]) and not args["multitask"],
                                                 C=kwargs["nlabels"], return_dists=args["boundary"])])
        elif args["softarget"]:
            print("NO FLIP IN AUGMENTATION, USING SOFTTARGET")
            selected += 1
            train_transforms = Compose([ReturnPatch(patch_size=(patch_size, patch_size), anyborder=args["anyborder"]),
                                        RandomAffine(), Intensity(), Noisify(), SoftTarget(p=1.0), ToTensor()])
        elif args["psoftarget"]:
            print("NO FLIP IN AUGMENTATION, USING PSOFTTARGET")
            selected += 1
            train_transforms = Compose([ReturnPatch(patch_size=(patch_size, patch_size), anyborder=args["anyborder"]),
                                        RandomAffine(), Intensity(), Noisify(), SoftTarget(p=0.2), ToTensor()])
        elif args["bordersoft"]:
            print("Only aug is border focus soft")
            selected += 1
            train_transforms = Compose([ReturnPatch(patch_size=(patch_size, patch_size), anyborder=args["anyborder"]),
                                        SoftTarget(p=1, border_focus=True), ToTensor()])
        elif args["affonly"]:
            print("Only aug is default affine")
            selected += 1
            train_transforms = Compose([ReturnPatch(patch_size=(patch_size, patch_size), anyborder=args["anyborder"]),
                                        RandomAffine(),
                                        ToTensor(transform_to_onehot=(args["gdl"] or args["boundary"]) and not args["multitask"],
                                                 C=kwargs["nlabels"], return_dists=args["boundary"])])
        elif args["intonly"]:
            print("Only aug is default intensity")
            selected += 1
            train_transforms = Compose([ReturnPatch(patch_size=(patch_size, patch_size), anyborder=args["anyborder"]),
                                        Intensity(),
                                        ToTensor(transform_to_onehot=(args["gdl"] or args["boundary"]) and not args["multitask"],
                                                 C=kwargs["nlabels"], return_dists=args["boundary"])])
        elif args["noiseonly"]:
            print("Only aug is default noisify")
            selected += 1
            train_transforms = Compose([ReturnPatch(patch_size=(patch_size, patch_size), anyborder=args["anyborder"]), Noisify(),
                                        ToTensor(transform_to_onehot=(args["gdl"] or args["boundary"]) and not args["multitask"],
                                                 C=kwargs["nlabels"], return_dists=args["boundary"])])
        elif args["center128"]:
            print("Baseline, center crop 128 no aug")
            selected += 1
            train_transforms = Compose([CenterCrop(128, 128), ToTensor()])
        elif args["center"]:
            print("Baseline, center crop 160 no aug")
            selected += 1
            train_transforms = Compose([CenterCrop(160, 160), ToTensor()])
        elif args["center_halfaug"]:
            print("Baseline, center crop 160 half aug")
            selected += 1
            train_transforms = Compose([CenterCrop(160, 160), RandomAffine(rotate=10, scale=(0.9, 1.1)),
                                        Intensity(brightness=0.05), Noisify(), ToTensor()])
        elif args["center_fullaug"]:
            print("Baseline, center crop 160 full aug")
            selected += 1
            train_transforms = Compose([CenterCrop(160, 160), RandomAffine(), Intensity(), Noisify(), ToTensor()])
        elif args["noaug"]:
            print("Only random {} patches".format(patch_size))
            selected += 1
            train_transforms = Compose([ReturnPatch(patch_size=(patch_size, patch_size), anyborder=args["anyborder"],
                                                    fullrandom=args["FULLRANDOM"]),
                                        ToTensor(transform_to_onehot=(args["gdl"] or args["boundary"]) and not args["multitask"],
                                                 C=kwargs["nlabels"], return_dists=args["boundary"])])
        elif args["sig5"]:
            print("Sigmoid target 5")
            selected += 1
            train_transforms = Compose([ReturnPatch(patch_size=(patch_size, patch_size)), SoftTarget(order=5, p=1), ToTensor()])
        elif args["sig10"]:
            print("Sigmoid target 10")
            selected += 1
            train_transforms = Compose([ReturnPatch(patch_size=(patch_size, patch_size)), SoftTarget(order=10, p=1), ToTensor()])
        elif args["sig15"]:
            print("Sigmoid target 15")
            selected += 1
            train_transforms = Compose([ReturnPatch(patch_size=(patch_size, patch_size)), SoftTarget(order=15, p=1), ToTensor()])
        elif args["sigaug"]:
            print("Sigmoid 15 with p 1")
            selected += 1
            train_transforms = Compose([ReturnPatch(patch_size=(patch_size, patch_size), anyborder=args["anyborder"],
                                                    fullrandom=args["FULLRANDOM"]), SoftTarget(order=15, p=1),
                                        ToTensor(transform_to_onehot=(args["gdl"] or args["boundary"]) and not args["multitask"],
                                                 C=kwargs["nlabels"], return_dists=args["boundary"])])
        elif args["sighalfaug"]:
            print("Halfaug + sig15 always")
            selected += 1
            train_transforms = Compose([ReturnPatch(patch_size=(patch_size, patch_size), anyborder=args["anyborder"],
                                                    fullrandom=args["FULLRANDOM"]),
                                        RandomAffine(rotate=10, scale=(0.9, 1.1)), Intensity(brightness=0.05), Noisify(),
                                        SoftTarget(order=15, p=1),
                                        ToTensor(transform_to_onehot=(args["gdl"] or args["boundary"]) and not args["multitask"],
                                                 C=kwargs["nlabels"], return_dists=args["boundary"])])
        else:
            print("Full default mnihip aug with random horflip")
            train_transforms = Compose([ReturnPatch(patch_size=(patch_size, patch_size), anyborder=args["anyborder"],
                                                    fullrandom=args["FULLRANDOM"]),
                                        RandomAffine(), Intensity(), RandomFlip(modes=['horflip']), Noisify(), ToTensor()])

        assert selected == 1, "error in augmentation selection. Selected: {}. check experiment {}".format(selected, args)
        data_transforms = {'train': train_transforms, 'validation': train_transforms,
                           'test': Compose([CenterCrop(160, 160),
                                            ToTensor(transform_to_onehot=(args["gdl"] or
                                                                          args["boundary"]) and not args["multitask"],
                                                     C=kwargs["nlabels"], return_dists=args["boundary"])])}

    else:  # volumetric transforms
        if args["hweights"]:
            print("USING HARP TUNED E2D WEIGHTS")
        print('Chaing transforms to volumetric transforms')
        crop_shape = (160, 160, 160)

        pre_saved_model = kwargs["pre_saved_model"]
        if args["e3d"]:
            totensor = ToTensor(volumetric=args["volumetric"], rune2d=True, e2d_device=kwargs["device"],
                                pre_saved_model=pre_saved_model, small=args["patch32"], C=kwargs["nlabels"],
                                return_dists=args["boundary"], addch=args["3CH"],
                                transform_to_onehot=(args["gdl"] or args["boundary"]) and not args["multitask"])
        else:
            pre_saved_model = None
            totensor = ToTensor(volumetric=args["volumetric"])

        test_transforms = Compose([CenterCrop(160, 160, 160), totensor])

        if args["patch32"]:
            crop_shape = (32, 32, 32)
            if args["aug3d"]:
                train_transforms = Compose([ReturnPatch(patch_size=crop_shape, kernel_shape=(3, 3),
                                            anyborder=args["anyborder"]), Intensity(), Noisify(), totensor], time_trans=False)
            else:
                train_transforms = Compose([ReturnPatch(patch_size=crop_shape, kernel_shape=(3, 3),
                                            anyborder=args["anyborder"]), totensor])
        elif args["aug3d"]:
            print("Adding augs to 3D transforms")
            train_transforms = Compose([CenterCrop(*crop_shape), Noisify(), totensor])
        else:
            train_transforms = test_transforms

        if args["mni"]:
            if not args["patch32"]:
                train_transforms.addto(MNITransform(), begin=True)
                print("MNI transform being applied to train transforms")
            else:
                print("Due to patch 32, not doing mni transform on train")
            print("MNI transform being applied to test transforms")
            test_transforms.addto(MNITransform(), begin=True)

        data_transforms = {'train': train_transforms,
                           'validation': train_transforms,
                           'test': test_transforms}

    print("\nUsed transforms")
    for k, v in data_transforms.items():
        print("{} {}".format(k, v), end='\n\n')

    return data_transforms


if __name__ == "__main__":
    print("To test transforms run dataset.py applying the transforms you want to test into the data")
