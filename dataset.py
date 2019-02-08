'''
Defines dataset class
Uses nift files and RAM caching with H5PY
Contains test functions for volume visualization and slice visualization

Author: Diedre Carmo
https://github.com/dscarmo
'''
import os
from os.path import join as add_path
import glob
from sys import argv
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import h5py
import nibabel as nib

import torch
import torch.utils.data as data
import torchvision
from transforms import ToTensor, ToFloat32, CenterCrop, Compose, Resize, RandomAffine, Intensity, RandomFlip, Noisify, ReturnPatch
from skimage.transform import rotate
from multiprocessing import Lock   
from utils import normalizeMri, viewnii

cla_lock = Lock()
adni_lock = Lock()

orientations = ["sagital", "coronal", "axial"] # original data orientations

default_datapath = os.path.join("/home","diedre","bigdata","mni_hip_data")
default_adni = os.path.join("/home","diedre","bigdata","manual_selection_rotated","isometric")

def unit_test(image_dataset=True, adni=True, shuffle=True, ntoshow=30, show=True, plt_show=True, nworkers=0, hiponly=True, volume=False, e2d=False):
    '''
    Tests vizualisation of a training batch of the dataset
    '''
    train_transforms = Compose([ReturnPatch(patch_size=(32, 32)), RandomAffine(), Intensity(), RandomFlip(modes=['horflip']), Noisify(), ToTensor()])
    #train_transforms = Compose((Resize(128, 128), RandomAffine(), ToTensor())) # for testing another stuff
    print("Testing all orientations...")
    for o in orientations:
        if adni:
            test = FloatHippocampusDataset(h5path=default_adni, mode="train", transform=train_transforms, data_split=(0.5, 0.1, 0.4), adni=True, orientation=o, hiponly=True, return_volume=False, e2d=True)
        else:
            test = FloatHippocampusDataset(mode="train", transform=train_transforms, orientation=o, hiponly=hiponly, return_volume=volume, e2d=e2d)

        test_loader =  data.DataLoader(test, batch_size=ntoshow, shuffle=shuffle, num_workers=0)
        batch = next(iter(test_loader))
        if show is True:
            display_batch(batch, o + " dataloader")
    if plt_show:
        plt.show()        
        
def display_batch(batch, title):
    '''
    Displays a batch content on a grid
    '''
    imgs, tgts = batch
    batch_len = len(imgs)
    grid_data = torch.zeros((batch_len, 1 , imgs.size(2), imgs.size(3)))
    
    for i, (im, tg) in enumerate(zip(imgs, tgts)):
        #Account for e2d
        if im.shape[0] == 3:
            overlap = (im[1]/2 + tg/2)
        else:
            overlap = (im/2 + tg/2)
        grid_data[i] = overlap
        
    grid = torchvision.utils.make_grid(grid_data, nrow=batch_len//5).numpy().transpose(1, 2 , 0)
    plt.figure(num=title)
    plt.title(str(batch_len) + " " + title + " samples")
    plt.axis('off')
    plt.imshow(grid)

def view_volumes(dataset_name="clarissa", wait=0):
    '''
    View volumes supplied by a dataset abstraction
    '''
    if dataset_name == "clarissa":
        fhd = FloatHippocampusDataset(hiponly=False, return_volume=True, orientation="coronal", mode="test")
    elif dataset_name == "cc359":
        fhd = CC359Data()
    elif dataset_name == "adni":
        #fhd = ADNI()
        fhd = FloatHippocampusDataset(h5path=default_adni, return_volume=True, mode="test", adni=True, data_split=(0.5, 0.1, 0.4))
    elif dataset_name == "simone":
        fhd = Simone()
    else:
        raise ValueError("Dataset {} not recognized".format(dataset_name))
    
    for i in range(len(fhd)):
        viewnii(*fhd[i], id=dataset_name, wait=wait)

class ADNI(data.Dataset):
    '''
    Abstracts ADNI data
    # Subjects ADNI: 002_S_0295, 002_S_0413, 002_S_0619, 002_S_0685, 002_S_0729, 002_S_0782, 002_S_0938, 002_S_0954, 002_S_0954, 002_S_1018, 002_S_1155, 003_S_0907, 003_S_0981, 005_S_0222, 005_S_0324, 005_S_0448, 005_S_0546, 005_S_0553, 005_S_0572, 005_S_0610, 005_S_0814, 005_S_1341, 006_S_0498, 006_S_0675, 006_S_0681, 006_S_0731, 006_S_1130, 007_S_0041, 007_S_0068, 007_S_0070, 007_S_0101, 007_S_0249, 007_S_0293, 007_S_0316, 007_S_0414, 007_S_0698, 007_S_1206, 007_S_1222, 007_S_1339, 009_S_0751, 009_S_1030, 011_S_0003, 011_S_0005, 011_S_0010, 011_S_0016, 011_S_0021, 011_S_0022, 011_S_0023, 011_S_0053, 011_S_0168
    '''
    def __init__(self, adnipath=default_adni):
        self.samples = glob.glob(adnipath + "/samples/*.nii.gz")
        self.adnipath = adnipath
        self.reconstruction_orientations = orientations
        if len(self.samples) == 0:
            raise ValueError("nii data not found, check if you have data in {}".format(adnipath))

    def __len__(self):
        return len(self.samples)
    
    def get_by_name(self, name):
        '''
        Name should be only id without format (without .nii)
        '''
        if name == "33":
            name = "33"
        return self.__getitem__(self.samples.index(add_path(self.adnipath, "samples", name + ".nii.gz")))

    def __getitem__(self, i):
        sample = self.samples[i]
        fname = os.path.basename(sample)

        volpath = sample
        hippath = add_path(self.adnipath, "masks", fname)   

        print("Subject {}".format(fname.split('.')[0]))
        vol = nib.load(volpath).get_fdata()
        hip = nib.load(hippath).get_fdata()
        vs = vol.shape
        hs = hip.shape
        
        if vs != hs:
            print("WARNING: shapes of mask and volume are not equal!")

        norm_vol = normalizeMri(vol.astype(np.float32))
        norm_hip = hip.astype(np.bool).astype(np.float32)
        
        return norm_vol, norm_hip


class Simone(data.Dataset):
    '''
    NOT IMPLEMENTED YET
    '''
    def __init__(self):
        pass

    def __len__(self):
        pass
    
    def __getitem__(self, i):
        pass


class FloatHippocampusDataset(data.Dataset):
    '''
    Using h5py float data
    '''
    def __init__(self, h5path="/home/diedre/bigdata/mni_hip_data", adni=False, orientation="coronal", mode="train", data_split=(0.8, 0.1, 0.1), transform=None, hiponly=False, return_volume=False, float16=True, e2d=False):
        '''
        Initializes dataset class
        h5path: path to h5 file containing all the data, in samples and masks groups containing a dataset for each volume
        orientation: one of sagital, coronal or axial
        mode: one of train, validation or test
        data_split: distribution of data over modes (train, val, test)
        transform: transforms to apply on slices. transforms should support 2 numpy arrays as input 
        hiponly: only return slices with hippocampus presence
        '''
        orientations = ["sagital", "coronal", "axial"]
        self.reconstruction_orientations = orientations
        modes = ["train", "validation", "test"]

        assert orientation in orientations, "orientation should be one of {}".format(orientations)
        assert mode in modes, "mode should be one of {}".format(modes)
        assert np.sum(np.array(data_split)) == 1.0, "data_split should sum to 1.0"
        self.file_lock = adni_lock if adni else cla_lock
        self.float16 = float16
        self.tofloat32 = ToFloat32()
        self.h5path = h5path
        self.orientation = orientation
        self.transform = transform
        self.return_volume = return_volume
        self.e2d = e2d
        self.adni = adni
        print("FHD slices dataset initialized using {}".format("adni" if adni else "clarissa"))
        
        self.volume_shape = (181, 217, 181) # only used for clarissa data
        if adni:
            self.adni_vols = ADNI()
            self.fname = os.path.join(h5path, "float16_adni_hip_data_hiponly.hdf5")
        elif hiponly:
            self.fname = os.path.join(h5path, "mni_hip_data_hiponly.hdf5")
        else:
            self.fname = os.path.join(h5path, "mni_hip_data_full.hdf5")
        
        
        self.file_lock.acquire()
        with h5py.File(self.fname) as h5file:
            samples = h5file["samples"][orientation]
            masks = h5file["masks"][orientation]

            self.ids = list(samples.keys())
        self.file_lock.release()

        train_split = int(data_split[0]*len(self.ids))
        val_split = train_split + int(data_split[1]*len(self.ids))

        if mode == "train":
            self.ids = self.ids[:train_split]
        elif mode == "validation":
            self.ids = self.ids[train_split:val_split]
        elif mode == "test":
            self.ids = self.ids[val_split:]

        self.volume_ids = []
        for i in self.ids:
            vid = i.split("_")[0]
            if vid == "42913":
                continue
            if vid not in self.volume_ids:
                self.volume_ids.append(vid)

    def __len__(self):
        '''
        Dataset size, measure by number of images on folder
        '''
        if self.return_volume:
            return len(self.volume_ids)    
        else:
            return len(self.ids)
    
    def get_volids(self):
        return self.volume_ids

    def get_by_name(self, id):
        return self.__getitem__(0, vid=id)

    def __getitem__(self, index, vid=None):
        '''
        Returns transformed image and mask (0-1 range, float)
        '''
        self.file_lock.acquire()
        with h5py.File(self.fname) as h5file:
            if self.return_volume: # return volume
                samples = h5file["samples"]["coronal"]
                masks = h5file["masks"]["coronal"]
                vi = self.volume_ids[index]
                if self.adni:
                    image, target = self.adni_vols.get_by_name(vi)
                else:
                    image = np.zeros(self.volume_shape, dtype=np.float32)
                    target = np.zeros(self.volume_shape, dtype=np.float32)
                    j = 0
                    for i in self.ids:
                        if vid is not None:
                            vi = vid
                        if i.split("_")[0] == vi:
                            # Rotate back to original volume orientation
                            image[:, j, :], target[:, j, :] = rotate(samples.get(i)[:], -90, resize=True), rotate(masks.get(i)[:], -90, resize=True)
                            j += 1  
                    print("Subject " + str(vi))  

            else: # return slice
                samples = h5file["samples"][self.orientation]
                masks = h5file["masks"][self.orientation]
                i = self.ids[index]

                if self.e2d:
                    preindex = index - 1
                    if preindex < 0:
                        preindex = index
                    postindex = index + 1
                    if postindex == len(self.ids):
                        postindex = index
                    pri = self.ids[preindex]
                    poi = self.ids[postindex]
                    
                    center_img, target = samples.get(i)[:], masks.get(i)[:]
                    image = np.zeros((3, center_img.shape[0], center_img.shape[1]), dtype=center_img.dtype)
                    
                    vol_i = i.split('_')[0]
                    if pri.split('_')[0] != vol_i: pri = i
                    if poi.split('_')[0] != vol_i: poi = i                        

                    image[0] = samples.get(pri)[:]
                    image[1] = center_img
                    image[2] = samples.get(poi)[:]
                else:
                    image, target = samples.get(i)[:], masks.get(i)[:]
                    
        self.file_lock.release()

        if self.float16 and not self.return_volume:
            image, target = self.tofloat32(image, target)

        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target

class CC359Data(data.Dataset):
    '''
    Abstracts public CC359 (https://sites.google.com/view/calgary-campinas-dataset/home)
    '''
    def __init__(self, cc359path="/home/diedre/bigdata/CC-359/Original", maskpath="/home/diedre/bigdata/CC-359/Hippocampus-segmentation/Automatic/hippodeep"):
        self.nift_paths = glob.glob(cc359path + "/*.nii.gz")
        self.mask_paths = glob.glob(maskpath + "/*mask.nii.gz")
        self.reconstruction_orientations = orientations

    def __len__(self):
        return len(self.nift_paths)
    
    def __getitem__(self, i):
        fid = os.path.basename(self.nift_paths[i])[:6]
        nift = nib.load(self.nift_paths[i])

        for mpath in self.mask_paths:
            if os.path.basename(mpath)[:6] == fid:
                mask_nift = nib.load(mpath)

        mdata = mask_nift.get_fdata()
        mnormalized = np.zeros((mdata.shape), dtype=np.float64)
        cv.normalize(mdata, mnormalized, alpha=1.0, beta=0.0, norm_type=cv.NORM_MINMAX)

        data = nift.get_fdata()
        normalized = np.zeros((data.shape), dtype=np.float64)
        cv.normalize(data, normalized, alpha=1.0, beta=0.0, norm_type=cv.NORM_MINMAX)
        
        return normalized.astype(np.float32), mnormalized.astype(np.float32)

def get_data(data_transforms, db="clarissa", datapath=default_datapath, nworkers=4, e2d=False,
             batch_size=20, test=False, hiponly=True, return_volume=False, shuffle = {"train": True, "validation": False, "test": False}):
    '''
    Abstracts acquiring dataloaders and data info for other classes
    Constructs dataloaders in all orientations
    Shuffles following shuffle dict
    Returns dataloaders, accessible by: dataloaders[orientation][mode]
    '''
    modes = ['train', 'validation', 'test']

    print("Using " + str(nworkers) + " processes")
    
    hips = {}
    if db == "adni":
        print("Getting dataloaders for ADNI slice data")
        for o in orientations:
            hips[o] = {m: FloatHippocampusDataset(h5path=default_adni, adni=True, data_split=(0.5, 0.1, 0.4), mode=m,  orientation=o, transform=data_transforms[m], hiponly=True, return_volume=False, e2d=e2d) for m in modes}
    elif db == "concat":
        print("Concatenating adni and Clarissa")
        for o in orientations:
            hips[o] = {m: data.ConcatDataset([FloatHippocampusDataset(h5path=default_adni, adni=True, data_split=(0.5, 0.1, 0.4), mode=m,  orientation=o, transform=data_transforms[m], hiponly=True, return_volume=False, e2d=e2d), 
                                              FloatHippocampusDataset(h5path=datapath, mode=m, orientation=o, transform=data_transforms[m], hiponly=True, return_volume=False, e2d=e2d)]) for m in modes}
    elif db == "clarissa":
        print("Using clarissa")
        for o in orientations:
            hips[o] = {m: FloatHippocampusDataset(h5path=datapath, mode=m, orientation=o, transform=data_transforms[m], hiponly=hiponly, return_volume=return_volume, e2d=e2d) for m in modes}
    else:
        raise ValueError("db {} not supported".format(db))
    
    hip_dataloaders = {}
    for o in orientations:
        hip_dataloaders[o] = {m: data.DataLoader(hips[o][m], batch_size=batch_size//int(19*(m=='test') + 1), shuffle=shuffle[m], num_workers=nworkers) for m in modes}

    print("dataset_sizes:")   
    dataset_sizes = {}         
    for o in orientations:
        hip_it = iter(hip_dataloaders[o]['train'])  # test iterator

        dataset_sizes[o] = {m: len(hips[o][m]) for m in modes}

        if test:
            display_batch(next(hip_it), o + " train dataloader")
    print(dataset_sizes)
    if test:
        plt.show() 

    return hip_dataloaders, dataset_sizes

def main():
    '''
    Runs if the module is called as a script (eg: python3 dataset.py <dataset_name> <frametime>)
    Executes self tests
    '''
    dataset_name = argv[1] if len(argv) > 1 else "clarissa"
    wait = argv[2] if len(argv) > 2 else 10
    print("Dataset module running as script, executing dataset unit test in {}".format(dataset_name))
    
    if dataset_name == "adni_slices":
        unit_test(image_dataset=False, adni=True, hiponly=True, plt_show=True, nworkers=4, e2d=True)
    elif dataset_name == "clarissa_slices":
        unit_test(image_dataset=False, adni=False, hiponly=True, plt_show=True, nworkers=4, e2d=True)
    elif dataset_name == "concat":
        from transforms import ReturnPatch, Intensity, RandomFlip, Noisify, ToTensor, CenterCrop, RandomAffine
        train_transforms = Compose([ReturnPatch(patch_size=(32, 32)), RandomAffine(), Intensity(), RandomFlip(modes=['horflip']), Noisify(), ToTensor()]) #default is 32 32 patch
        data_transforms = {'train': train_transforms, 'validation': train_transforms, 'test': Compose([CenterCrop(160, 160), ToTensor()])}
        mode = "train"
        data, dsizes = get_data(data_transforms=data_transforms, db="concat", e2d=True, batch_size=50 + 150*(mode!="test"))
        print("Dataset sizes: {}".format(dsizes))
        for o in orientations:
            batch = next(iter(data[o][mode]))
            display_batch(batch, o + " concat " + mode + " data")
        plt.show()
    else:
        view_volumes(dataset_name, wait=10)
    print("All tests completed!")

if __name__ == "__main__":            
    main()