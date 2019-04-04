'''
Contains custom transform functions supporting a image and a mask as io
To ensure compatibility input should be always numpy inside range 0-1, and float32, excpet for ToFloat32

#TODO adapt transforms to transform 3d volumes

Author: Diedre Carmo
https://github.com/dscarmo
'''
import time
import torch
import torch.nn.functional as F
from math import inf
import numpy as np
import cv2 as cv
import nibabel as nib
import random
import datetime
from tqdm import tqdm
import sparse as sparse3d
from scipy.sparse import dok_matrix
from skimage.transform import rotate
from utils import myrotate, normalizeMri, get_slice, get_device
from label import get_largest_components

class Compose(object):
    '''
    Executes all transforms given in tranlist (as a list)
    '''
    def __init__(self, tranlist, time_trans=False):
        self.tranlist = tranlist
        self.time_trans = time_trans
    
    def __call__(self, img, mask):
        for tran in self.tranlist:
            if self.time_trans:
                begin = time.time()
            img, mask = tran(img, mask)
            if self.time_trans:
                print("{} took {}s".format(tran, time.time() - begin))
        if self.time_trans: print("-------- Composed Transforms Finished ---------")
        return img, mask

    def __str__(self):
        string = ""
        for tran in self.tranlist:
            string += str(tran) + ' '
        return string[:-1]

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
        shape = img.shape
        len_shape = len(shape)

        cropx = self.cropx
        cropy = self.cropy
        cropz = self.cropz
        
        if len_shape > 2:
            z, y, x = img.shape
        else:
            y, x = img.shape

        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)   
        if self.volumetric:
            startz = z//2-(cropz//2)
            rimg, rmask = img[startz:startz+cropz, starty:starty+cropy, startx:startx+cropx], mask[startz:startz+cropz, starty:starty+cropy, startx:startx+cropx]
            return rimg, rmask
        else:
            if len_shape > 2: 
                return img[:, starty:starty+cropy,startx:startx+cropx], mask[starty:starty+cropy,startx:startx+cropx]
            else:
                return img[starty:starty+cropy,startx:startx+cropx], mask[starty:starty+cropy,startx:startx+cropx]

    def __str__(self):
        return "CenterCrop, patch size: {}x{}x{} volumetric {}".format(self.cropx, self.cropy, self.cropz, self.volumetric)

class ToTensor(object):
    '''
    Convert ndarrays in sample to Tensors.
    '''
    def __init__(self, volumetric=False, rune2d=False, e2d_device=None, pre_saved_model=None, small=False):
        self.volumetric = volumetric
        if rune2d:
            assert self.volumetric and pre_saved_model is not None, "to rune2d in ToTensor, you need volumetric input and a pre_saved_model!"
        self.pre_saved_model = pre_saved_model
        self.rune2d = rune2d
        if e2d_device is None and self.rune2d:
            self.e2d_device = get_device(verbose=False)
        else:
            self.e2d_device = e2d_device
        self.small = small

    def __call__(self, npimage, npmask):
        '''
        input numpy image: H x W
        output torch image: C X H X W
        '''
        if self.rune2d:
            ishape = (2, npimage.shape[0], npimage.shape[1], npimage.shape[2])
            image = torch.zeros(ishape).float()
            np_input, np_e2d_output = run_once(None, self.pre_saved_model, numpy_input=npimage, save=False, verbose=False, device=self.e2d_device, small=self.small)
            image[0] = torch.from_numpy(np_input).float()
            image[1] = torch.from_numpy(np_e2d_output).float()
        elif len(npimage.shape) == 2 or (len(npimage.shape) == 3 and self.volumetric):
            image = torch.unsqueeze(torch.from_numpy(npimage), 0).float()
        else:
            image = torch.from_numpy(npimage).float()
        mask = torch.unsqueeze(torch.from_numpy(npmask), 0).float()
        image[image > 1.0] = 1.0
        image[image < 0.0] = 0.0
        mask[mask > 1.0] = 1.0
        mask[mask < 0.0] = 0.0
        return image, mask

    def __str__(self):
        return "ToTensor, volumetric: {}, rune2d: {}, e2d_device: {}, pre_saved_model passed: {}, small: {}".format(self.volumetric, self.rune2d, self.e2d_device, self.pre_saved_model is not None, self.small)


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
    #TODO collate problem
    '''
    def __init__(self, ppositive=0.8, patch_size=(32, 32), kernel_shape=(3,3)):
        '''
        Sets desired patchsize (width, height)
        '''
        self.psize = patch_size
        self.ppositive = ppositive
        self.kernel = np.ones(kernel_shape, np.uint8)
        self.ks = kernel_shape
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
        random.seed()
        # Get list of candidates for patch center
        e2d = False
        shape = image.shape
        if not self.volumetric and len(shape) == 3:
            shape = (shape[1], shape[2])
            e2d = True
        
        if self.volumetric:
            borders = np.zeros(shape, dtype=mask.dtype)
            for i in range(shape[0]):
                uintmask = (mask[i]*255).astype(np.uint8)
                borders[i] = ((uintmask - cv.erode(uintmask, self.kernel, iterations=1))/255).astype(mask.dtype)
            sparse = sparse3d.COO.from_numpy(borders)
            keylist = sparse.nonzero()
        else:
            # Get border of mask
            uintmask = (mask*255).astype(np.uint8)
            borders = ((uintmask - cv.erode(uintmask, self.kernel, iterations=1))/255).astype(mask.dtype)
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
        if len(keylist[0]) > 0 and random.random() < self.ppositive:
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
                tl_rrow, tl_rcol, tl_rdepth = random.randint(0, tl_row_limit), random.randint(0, tl_col_limit), random.randint(0, tl_depth_limit)
            else:
                tl_rrow, tl_rcol = random.randint(0, tl_row_limit), random.randint(0, tl_col_limit)
            
        if tl_rrow < 0: 
            tl_rrow = 0
        if tl_rcol < 0: 
            tl_rcol = 0
        if self.volumetric:
            if tl_rdepth < 0: tl_rdepth = 0

        if debug:
            print("Patch top left(row, col): {} {}".format(tl_rrow, tl_rcol))     

        if self.volumetric:
            rimage, rmask = image[tl_rrow:tl_rrow + self.psize[0], tl_rcol:tl_rcol + self.psize[1], tl_rdepth:tl_rdepth + self.psize[2]], mask[tl_rrow:tl_rrow + self.psize[0], tl_rcol:tl_rcol + self.psize[1], tl_rdepth:tl_rdepth + self.psize[2]]
            assert rimage.shape == self.psize and rmask.shape == self.psize, "fatal error generating patches, incorrect patch size image: {}, mask: {}, intended patch size: {}".format(rimage.shape, rmask.shape, self.psize)
            return rimage, rmask
        else:
            if e2d:
                return image[:, tl_rrow:tl_rrow + self.psize[0], tl_rcol:tl_rcol + self.psize[1]], mask[tl_rrow:tl_rrow + self.psize[0], tl_rcol:tl_rcol + self.psize[1]]
            else:
                return image[tl_rrow:tl_rrow + self.psize[0], tl_rcol:tl_rcol + self.psize[1]], mask[tl_rrow:tl_rrow + self.psize[0], tl_rcol:tl_rcol + self.psize[1]]
    def __str__(self):
        return "ReturnPatch: ppositive {} patch_size {}, kernel_shape {}, volumetric {}".format(self.ppositive, self.psize, self.ks, self.volumetric)

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
        if random.random() < self.p:
            e2d = False
            if len(image.shape) == 3:
                e2d = True
            op = random.choice(self.modes)
            if e2d:
                if op == 'horflip':
                    mask = np.fliplr(mask).copy()
                    for i in range(3):
                        image[i] = np.fliplr(image[i]).copy()
                elif op == 'verflip':
                    mask = np.flipud(mask).copy()
                    for i in range(3):
                        image[i] = np.flipud(image[i]).copy()
                elif op == 'both':
                    mask = np.fliplr(np.flipud(mask)).copy()
                    for i in range(3):
                        image[i] = np.fliplr(np.flipud(image[i])).copy()
                else:
                    raise ValueError("Invalid RandomFlip mode: {}".format(op)) 
            else:
                if op == 'horflip':
                    image, mask = np.fliplr(image).copy(), np.fliplr(mask).copy()
                elif op == 'verflip':
                    image, mask = np.flipud(image).copy(), np.flipud(mask).copy()
                elif op == 'both':
                    image, mask = np.fliplr(np.flipud(image)).copy(), np.fliplr(np.flipud(mask)).copy()
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
    def __init__(self, p=1, brightness=0.10):
        assert p > 0 and p <= 1 and brightness > 0 and brightness <= 1, "arguments make no sense"
        self.p = p
        self.b = brightness

    def __call__(self, image, mask):
        if random.random() < self.p:
            value = ((random.random() - 0.5)*2)*self.b
            image += value
            
        return image, mask

    def __str__(self):
        return "Intensity: p {}, brightness {}".format(self.p, self.b)

class Noisify(object):
    '''
    Randomly adds gausian noise
    '''
    def __init__(self, p=0.2, var=0.0002, mean=0):    
        self.p = p
        self.var = var
        self.mean = mean
        self.sigma = var**0.5
    
    def __call__(self, image, mask):
        if random.random() < self.p:
            shape = image.shape
            gauss = np.random.normal(self.mean, self.sigma, shape)
            gauss = gauss.reshape(*shape)
            inoisy = image + gauss
            return inoisy, mask
        else:
            return image, mask

    def __str__(self):
        return "Noisify: p {}, var {}, mean {}, sigma {}".format(self.p, self.var, self.mean, self.sigma)

class RandomAffine(object):
    '''
    Randomly applies scale -> rotation to simulate different data source
    WARNING: Input has to have even shape, will be fixed soon
    TODO: Volumetric support
    '''
    LESS = -1
    EQUAL = 0
    MORE = 1
    def __init__(self, p=0.2, rotate=20, scale=(0.8, 1.2), fillmode='symmetric', volumetric=False):
        self.p = p
        self.rotate = rotate
        self.scale = scale
        self.fmode = fillmode
        self.volumetric = volumetric #TODO
        
    def __call__(self, image, mask, debug=False):
        if random.random() < self.p:
            # Save original size for final center crop
            original_size = np.array(image.shape)
            
            e2d = False

            # Adapt shape to E2D
            if len(original_size) == 3: #E2D
                e2d = True
                original_size = np.array((original_size[1], original_size[2]))
            self.center_croper = CenterCrop(original_size[1], original_size[0])
            if debug:
                print("Original size {}".format(original_size))
                print("E2D {}".format(e2d))
            # Pick a scale factor
            scale = random.uniform(self.scale[0], self.scale[1])
            
            # Target scale size
            scale_size = (scale*original_size).astype(np.int)
            
            # If odd, add + 1 for simplicity of padding later
            if scale_size[0]%2: 
                scale_size += 1
            
            # Apply scale
            self.resizer = Resize(scale_size[1], scale_size[0])
            image, mask = self.resizer(image, mask)
            if debug:
                print("After resize {}".format(image.shape))
            # Apply rotation
            ang = 2*(random.random() - 0.5)*self.rotate
            if debug:
                print("Rotating by {}".format(ang))
            if e2d:
                for i in range(3):
                    image[i] = rotate(image[i], ang, preserve_range=True, mode='constant', order=3)
            else:
                image = rotate(image, ang, preserve_range=True, mode='constant', order=3)
            mask = rotate(mask, ang, preserve_range=True, mode='constant', order=3)
            
            if debug:
                print("After rot shape {}".format(image.shape))

            # Go back to original size
            shape_diff, (rowdiff, coldiff) = self.compare_shapes(image.shape, original_size)
            if shape_diff == self.LESS:
                if e2d:
                    buffer = np.zeros((3, original_size[0], original_size[1]), dtype=image.dtype)
                    for i in range(3):
                        buffer[i] = np.pad(image[i], (rowdiff//2, coldiff//2), mode=self.fmode)
                    image = buffer
                else:
                    image = np.pad(image, (rowdiff//2, coldiff//2), mode=self.fmode)
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
        return "RandomAffine: p {}, rotate {}, scale {}, fmode {}, volumetric {}".format(self.p, self.rotate, self.scale, self.fmode, self.volumetric)

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
        self.volumetric = volumetric #TODO

    def __call__(self, image, mask):
        '''
        Image and mask should be numpy 2D arrays or E2D
        '''
        if len(image.shape) == 3:
            nimage = np.zeros((3, self.size[1], self.size[0]),dtype=image.dtype)
            for i in range(3):
                nimage[i] =  cv.resize(image[i], self.size, interpolation=cv.INTER_CUBIC)
            mask = cv.resize(mask, self.size)
        else:
            nimage = cv.resize(image, self.size, interpolation=cv.INTER_CUBIC)
            mask = cv.resize(mask, self.size, interpolation=cv.INTER_NEAREST)

        return nimage, mask
    
    def __str__(self):
        return "Resize: size {}, volumetric {}".format(self.size, self.volumetric)


def run_once(volpath, models, numpy_input=None, save=True, verbose=True, device=None, small=False, return_mask_path=False):
    '''
    Runs our best model in a provided volume and saves mask,
    In a self contained matter
    To serve a ready (normalized) numpy_input, volpath should be None
    '''
    assert volpath is not None or numpy_input is not None, "volpath or numpy input should not be None"
    if verbose:
        print("\nWARNING: For the best performance, the provided volume should return slices on the following way for optimal performance:")
        print("volume[0, :, :] sagital, eyes facing down")
        print("volume[:, 0, :] coronal")
        print("volume[:, :, 0] axial, with eyes facing right")
        print("Those are MNI register orientations. We also recommend registering to MNI for the best performance (available with command -reg if you have FSL installed).\n")
    begin = time.time()
    
    if device is None:
        device = get_device()
    orientations = ["sagital", "coronal", "axial"] 
    if small:
        CROP_SHAPE = 32
    else:
        CROP_SHAPE = 160
    slice_transform = Compose([CenterCrop(CROP_SHAPE, CROP_SHAPE), ToTensor()])
    
    if numpy_input is not None:
        sample_v = numpy_input
    else:
        sample_v = normalizeMri(nib.load(volpath).get_fdata().astype(np.float32))

    shape = sample_v.shape
    sum_vol_total = torch.zeros(shape)

    for o, model in models.items():
        model.eval()
        model.to(device)

    if verbose: print("Performing segmentation...")
    for i, o in enumerate(orientations):
        if verbose: print("Processing {} view...".format(o))
        slice_shape = myrotate(get_slice(sample_v, 0, o), 90).shape
        for j in tqdm(range(shape[i])):
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
                    
            s, _ = slice_transform(ts, ts[1]) # work around, no mask
            s = s.to(device)
            
            probs = models[o](s.unsqueeze(0))
            
            cpup = probs.squeeze().detach().cpu()
            finalp = torch.from_numpy(myrotate(cpup.numpy(), -90)).float() # back to volume orientation

            # Add to final consensus volume, uses original orientation/shape
            if i == 0:
                toppad = shape[1]//2 - CROP_SHAPE//2
                sidepad = shape[2]//2 - CROP_SHAPE//2
                
                tf = 1 if shape[1]%2 == 1 else 0
                sf = 1 if shape[2]%2 == 1 else 0
                pad = F.pad(finalp, (sidepad + sf, sidepad, toppad, toppad + tf))/3

                sum_vol_total[j, :, :] += pad
            elif i == 1:
                toppad = shape[0]//2 - CROP_SHAPE//2
                sidepad = shape[2]//2 - CROP_SHAPE//2

                tf = 1 if shape[0]%2 == 1 else 0
                sf = 1 if shape[2]%2 == 1 else 0
                pad = F.pad(finalp, (sidepad+ sf, sidepad , toppad, toppad + tf))/3

                sum_vol_total[:, j, :] += pad
            elif i == 2:
                toppad = shape[0]//2 - CROP_SHAPE//2
                sidepad = shape[1]//2 - CROP_SHAPE//2

                tf = 1 if shape[0]%2 == 1 else 0
                sf = 1 if shape[1]%2 == 1 else 0
                pad = F.pad(finalp, (sidepad + sf, sidepad, toppad, toppad + tf))/3

                sum_vol_total[:, :, j] += pad

    if verbose: print("Performing consensus...")
    final_nppred = get_largest_components(sum_vol_total.numpy(), mask_ths=0.5)
    if verbose: print("Consensus done.")

    if save:
        if volpath is not None:
            save_path = volpath + "_voxelcount-{}_e2dhipmask.nii.gz".format(int(final_nppred.sum()))
        else:
            save_path = datetime.datetime.now().isoformat() + "_voxelcount-{}_e2dhipmask.nii.gz".format(int(final_nppred.sum()))
        
        nib.save(nib.nifti1.Nifti1Image(final_nppred, None), save_path)
    
    if verbose: print("Finished segmentation in {}s".format(time.time() - begin))
        
    if return_mask_path:
        return sample_v, final_nppred, save_path
    else:
        return sample_v, final_nppred

if __name__ == "__main__":
    print("To test transforms run dataset.py applying the transforms you want to test into the data")