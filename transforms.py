'''
Contains custom transform functions supporting a image and a mask as io
To ensure compatibility input should be always numpy inside range 0-1, and float32, excpet for ToFloat32

Author: Diedre Carmo
https://github.com/dscarmo
'''
import time
import torch
from math import inf
import numpy as np
import cv2 as cv
import random
from scipy.sparse import dok_matrix
from skimage.transform import rotate


class Compose(object):
    '''
    Executes all transforms given in tranlist (as a list)
    '''
    def __init__(self, tranlist):
        self.tranlist = tranlist
    
    def __call__(self, img, mask):
        for tran in self.tranlist:
            img, mask = tran(img, mask)
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
    def __init__(self, cropx, cropy):
        '''
        cropx: crop width
        cropy: crop height
        '''
        self.cropx = cropx
        self.cropy = cropy

    def __call__(self, img, mask):
        '''
        img: 2D numpy array
        mask: 2D numpy array
        '''
        shape = img.shape
        len_shape = len(shape)

        cropx = self.cropx
        cropy = self.cropy

        if len_shape > 2:
            _, y,x = img.shape
        else:
            y,x = img.shape

        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)   
        if len_shape > 2: 
            return img[:, starty:starty+cropy,startx:startx+cropx], mask[starty:starty+cropy,startx:startx+cropx]
        else:
            return img[starty:starty+cropy,startx:startx+cropx], mask[starty:starty+cropy,startx:startx+cropx]


class ToTensor(object):
    '''
    Convert ndarrays in sample to Tensors.
    '''
    def __call__(self, npimage, npmask):
        '''
        input numpy image: H x W
        output torch image: C X H X W
        '''
        if len(npimage.shape) == 2:
            image = torch.unsqueeze(torch.from_numpy(npimage), 0).float()
        else:
            image = torch.from_numpy(npimage).float()
            
        mask = torch.unsqueeze(torch.from_numpy(npmask), 0).float()
        image[image > 1.0] = 1.0
        image[image < 0.0] = 0.0
        mask[mask > 1.0] = 1.0
        mask[mask < 0.0] = 0.0

        
        return image, mask

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

class ReturnPatch(object):
    '''
    Random patch centered around hippocampus 
    If no hippocampus present, random patch
    Ppositive is chance of returning a patch around the hippocampus
    Kernel shape is shape of kernel for boundary extraction
    '''
    def __init__(self, ppositive=0.8, patch_size=(32, 32), kernel_shape=(3,3)):
        '''
        Sets desired patchsize (width, height)
        '''
        self.psize = patch_size
        self.ppositive = ppositive
        self.kernel = np.ones(kernel_shape, np.uint8)

    def __call__(self, image, mask, debug=False):
        '''
        Returns patch of image and mask
        '''
        # Get list of candidates for patch center
        e2d = False
        shape = image.shape
        if len(shape) == 3:
            shape = (shape[1], shape[2])
            e2d = True

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
        tl_rrow = inf
        tl_rcol = inf
        if len(keylist) > 0 and random.random() < self.ppositive:
            while tl_rrow > tl_row_limit or tl_rcol > tl_col_limit:
                tl_rrow, tl_rcol = random.choice(list(sparse.keys()))
                tl_rrow -= self.psize[0]//2 
                tl_rcol -= self.psize[1]//2
        else:
            tl_rrow, tl_rcol = random.randint(0, tl_row_limit), random.randint(0, tl_col_limit)

        if debug:
            print("Patch top left(row, col): {} {}".format(tl_rrow, tl_rcol))     

        if e2d:
            return image[:, tl_rrow:tl_rrow + self.psize[0], tl_rcol:tl_rcol + self.psize[1]], mask[tl_rrow:tl_rrow + self.psize[0], tl_rcol:tl_rcol + self.psize[1]]
        else:
            return image[tl_rrow:tl_rrow + self.psize[0], tl_rcol:tl_rcol + self.psize[1]], mask[tl_rrow:tl_rrow + self.psize[0], tl_rcol:tl_rcol + self.psize[1]]


class ToFloat32(object):
    '''
    Transforms stored float16 to float32
    '''
    def __call__(self, image, mask):
        return image.astype(np.float32), mask.astype(np.float32)


class RandomFlip(object):
    '''
    Randomly flip vertically, or horizontally
    1-p % chance to not do anything
    if doing something, sort between modes=( horflip, verflip, both)
    change modes variable to remove any of these or chance
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

     
class RandomAffine(object):
    '''
    Randomly applies scale -> rotation to simulate different data source
    WARNING: Input has to have even shape, will be fixed soon
    '''
    LESS = -1
    EQUAL = 0
    MORE = 1
    def __init__(self, p=0.2, rotate=20, scale=(0.8, 1.2), fillmode='symmetric'):
        self.p = p
        self.rotate = rotate
        self.scale = scale
        self.fmode = fillmode
        
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
                    image[i] = rotate(image[i], ang, preserve_range=True, mode='constant')
            else:
                image = rotate(image, ang, preserve_range=True, mode='constant')
            mask = rotate(mask, ang, preserve_range=True, mode='constant')
            
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
    Resize sample and mask
    '''
    def __init__(self, width, height):
        '''
        size: (width, height)
        '''
        self.size = (width, height)

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
if __name__ == "__main__":
    print("To test transforms run dataset.py applying the transforms you want to test into the data")