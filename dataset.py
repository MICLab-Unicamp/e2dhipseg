'''
Defines dataset classes for various used datasets, some are not used anymore
Contains test functions for volume visualization and slice visualization

Author: Diedre Carmo
https://github.com/dscarmo
'''
import os
from os.path import join as add_path
import glob
import pickle
from sys import argv
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2 as cv
import h5py
import collections
import nibabel as nib
import psutil
import time
import copy
import json
import torch
import torch.utils.data as data
from torch.utils.data import ConcatDataset
import torchvision
from transforms import ToTensor, ToFloat32, Compose, RandomAffine, Intensity, Noisify, SoftTarget, CenterCrop
import multiprocessing as mp
from multiprocessing import Lock, Process, Queue, Manager
from utils import normalizeMri, viewnii, myrotate, int_to_onehot, chunks, HALF_MULTI_TASK_NCHANNELS, MULTI_TASK_NCHANNELS
from utils import half_multi_task_labels, limit_multi_labels, imagePrint, type_assert, one_hot, get_slice, ITKManager, split_l_r
from nathip import NatHIP, get_group

cla_lock = Lock()
adni_lock = Lock()

orientations = ["sagital", "coronal", "axial"]  # original data orientations

DEFAULT_PATH = "../data"
VALID_MODES = ['train', 'validation', 'test']

# Post migration paths
default_datapath = os.path.join("/home", "diedre", "Dropbox", "bigdata", "mni_hip_data")
default_adni = os.path.join("/home", "diedre", "Dropbox", "bigdata", "manual_selection_rotated", "isometric")
mni_adni = os.path.join("/home", "diedre", "Dropbox", "bigdata", "manual_selection_rotated", "raw2mni")
mni_harp = os.path.join("/home", "diedre", "Dropbox", "bigdata", "harp", "mniharp")
default_harp = os.path.join("/home", "diedre", "Dropbox", "bigdata", "harp")
multitask_path = os.path.join("/home", "diedre", "Dropbox", "bigdata", "Hippocampus", "volbrain", "PACIENTES_E_CONTROLES")
multitask_hip_processed = os.path.join("/home", "diedre", "Dropbox", "bigdata", "Hippocampus", "processed")
multitask_hip_processed_slices = os.path.join("/home", "diedre", "Dropbox", "bigdata", "Hippocampus", "processed_slices")


HARP_CLASSES = ["cn", "mci", "ad"]


class DTISegDataset(data.Dataset):
    '''
    Abstracts segmentation with DTI data
    '''
    def __init__(self, mode, path=DEFAULT_PATH, transform=None, verbose=True, orientation=None, zero_background=True,
                 balance="balanced_205", norm_type="zero_to_plus", split=(0.8, 0.2), overhide_folder_name=None,
                 limit_masks=False, patch_size=64, register_strategy='v01', use_t1=False, displasia=False, t1_separate=False):
        '''
        path: folder containing data
        mode: one of ['train', 'validation', 'test'], test will return volumes, other will return patches
        transform: list of transforms to apply
        verbose: wether to print a lot of stuff or not
        orientation: one of ['sagital', 'coronal', 'axial'] or None if using test mode
        nlr: wether to return left and right labels
        display_samples: if different than 0, displays number of samples given
        '''
        super(DTISegDataset, self).__init__()
        assert mode in VALID_MODES, "mode {} should be one of {}".format(mode, VALID_MODES)
        assert orientation in orientations or orientation is None, "orientation {} should be one of {}".format(orientation,
                                                                                                               orientations)
        if mode == 'test':
            assert orientation is None, "test mode does not support orientation other than None"
            assert balance == "test", "in test mode, balance does not matter, use balance='test'"
        assert norm_type in ["zero_to_plus", "minus_to_plus", "mixed"], "norm type {} not support".format(norm_type)
        assert balance in ["test", "2020", "205", "51010", "355", "51515", "510"], "norm type {} not support".format(balance)
        assert np.array(split).sum() == 1, "split makes no sense, should sum to 1"
        assert patch_size in [32, 64]
        assert register_strategy in ["v01", "v02"]
        assert use_t1 in [False, "t1only", "t1dti"]

        self.use_t1 = use_t1
        self.mode = mode
        self.orientation = orientation
        self.zero_background = zero_background
        self.transform = transform
        self.verbose = verbose
        self.limit_masks = limit_masks
        self.displasia = displasia
        self.t1_separate = t1_separate

        separator = os.sep
        path_tokens = path.split(separator)

        folder_name = None
        if overhide_folder_name is not None:
            folder_name = overhide_folder_name
        else:
            if self.displasia:
                if mode == "test":
                    folder_name = "01"
                elif balance == "510":
                    if norm_type == "zero_to_plus":
                        if orientation == "sagital":
                            folder_name = "01"
                        elif orientation == "coronal":
                            folder_name = "02"
                        elif orientation == "axial":
                            folder_name = "03"
            else:
                if register_strategy == "v01":
                    if balance == "2020":
                        folder_name = "00"
                    elif balance == "205" or balance == "test":
                        if norm_type == "zero_to_plus":
                            folder_name = "01"
                        elif norm_type == "minus_to_plus":
                            folder_name = "02"
                        elif norm_type == "mixed":
                            folder_name = "03"
                    elif norm_type == "zero_to_plus":
                        if patch_size == 32:
                            if balance == "51010":
                                folder_name = "04"
                            elif balance == "355":
                                folder_name = "05"
                        elif patch_size == 64:
                            if balance == "51010":
                                folder_name = "06"
                            elif balance == "355":
                                folder_name = "07"
                elif register_strategy == "v02":
                    if balance == "test":
                        folder_name = "04"
                    elif norm_type == "zero_to_plus":
                        if patch_size == 64:
                            if balance == "355":
                                folder_name = "16"
                            elif balance == "51010":
                                folder_name = "18"
                            elif balance == "51515":
                                if orientation == "sagital":
                                    if limit_masks:
                                        folder_name = "23"
                                    else:
                                        # folder_name = "19"
                                        folder_name = "20"
                                elif orientation == "coronal":
                                    folder_name = "21"
                                elif orientation == "axial":
                                    folder_name = "22"
                        elif patch_size == 32:
                            if balance == "355":
                                folder_name = "14"
                            elif balance == "51010":
                                folder_name = "15"

        if folder_name is None:
            raise ValueError("Unsupported combination of patch_size, balance, norm_type and register_strategy: "
                             "{} {} {} {}".format(patch_size, balance, norm_type, register_strategy))

        self.folder_name = folder_name

        if self.displasia:
            if self.mode == "test":
                pre_folder = "Displasia/test"
            else:
                pre_folder = "Displasia/patches"
        elif self.mode == "test":
            pre_folder = "TestData"
        else:
            pre_folder = "patches"

        if path_tokens[0] == "..":  # Work around relative pathing
            glob_args = [os.path.dirname(os.getcwd())] + path_tokens[1:] + [pre_folder, folder_name, "*.npz"]
            self.items = glob.glob(os.path.join(*glob_args))
        else:
            glob_args = (path, pre_folder, folder_name, "*.npz")
            self.items = glob.glob(os.path.join(*glob_args))

        glob_args[-1] = "*.txt"
        try:
            print(glob_args)
            readme_path = glob.glob(os.path.join(*glob_args))[0]
        except IndexError:
            print("Readme file for dataset not found.")

        data_folder = os.path.join(*glob_args[:-1])

        if self.mode != "test":
            print(os.path.join(data_folder, self.mode + ".pkl"))
            if os.path.isfile(os.path.join(data_folder, self.mode + ".pkl")):
                with open(os.path.join(data_folder, self.mode + '.pkl'), 'rb') as saved_items:
                    self.items = pickle.load(saved_items)
                for i, v in enumerate(self.items):
                    self.items[i] = os.path.join(data_folder, os.path.basename(self.items[i]))  # support different folders
            else:
                print("PKL items file not saved, creating new ones...")
                stop_point = int(len(self.items)*split[0])
                print("Dividing dataset in point: {}".format(stop_point))
                with open(os.path.join(data_folder, 'train.pkl'), 'wb') as to_save_items:
                    pickle.dump(self.items[:stop_point], to_save_items)
                with open(os.path.join(data_folder, 'validation.pkl'), 'wb') as to_save_items:
                    pickle.dump(self.items[stop_point:], to_save_items)
                if self.mode == "train":
                    self.items = self.items[:stop_point]
                elif self.mode == "validation":
                    self.items = self.items[stop_point:]

        print("DTISegDataset initialized with nitems: {}, mode: {}, path: {}, transform: {}, "
              "orientation: {}, zero_background: {}, "
              "balance: {}, norm_type: {}, limit_masks: {}"
              "folder_name: {}".format(len(self.items), mode, path, transform, orientation,
                                       zero_background, balance, norm_type, limit_masks, folder_name))

        with open(readme_path) as readme_file:
            readme = readme_file.read()
            print(('-'*20 + "\nREADME: {}\n" + '-'*20).format(readme))

    def __len__(self):
        '''
        Returns number of items in the dataset
        '''
        return len(self.items)

    def __getitem__(self, i):
        '''
        Returns input data and target
        '''
        if self.verbose:
            print("Dataset returning {}".format(self.items[i]))
        npz = np.load(self.items[i])
        if self.mode == "test":
            dti, target, t1 = (npz["DTI_measures"], npz["mask_onehot"], npz["T1"])
        else:
            dti, target, t1 = (npz["DTI_measures"], npz[(self.mode != "test")*"patch_" + "mask_onehot"],
                               npz[(self.mode != "test")*"patch_" + "T1"])

        if self.displasia:
            t2 = npz["test_T2"] if self.mode == "test" else npz["patch_T2"]

        if self.use_t1 == "t1only":
            data = np.zeros((1,) + t1.shape, dtype=t1.dtype)
            data[0] = t1
        elif self.use_t1 == "t1dti":
            data = np.zeros((5,) + t1.shape, dtype=t1.dtype)
            data[:4] = dti
            data[4] = t1
        elif self.displasia:
            data = np.zeros((3,) + t1.shape, dtype=t1.dtype)
            data[0] = dti
            data[1] = t1
            data[2] = t2
        else:
            data = dti

        data = data.astype(np.float32)
        target = target.astype(np.float32)

        if self.zero_background:
            target[0] = 0

        if self.limit_masks is True:
            # Deprecated, causes negative inbalance
            buffer = np.zeros((4,) + target.shape[1:], dtype=target.dtype)
            buffer[:4] = target[1:5]
            # buffer[4] = target[6]  # removed hip
            target = buffer

        if self.mode == 'test':
            target[0] = 0
            target[5:9] = 0

        if self.transform is not None:
            data, target = self.transform(data, target)

        if self.mode == 'test':
            return os.path.basename(self.items[i]).split('.')[0], data, target
        else:
            return data, target

    def get_dataloader(self, batch_size, shuffle, nworkers=0):
        '''
        batch_size: batch size
        shuffle: follow dataset order or return randomized items
        nworkers: number of processes to use
        '''
        return data.DataLoader(self, batch_size, shuffle, num_workers=nworkers)

    def display_samples(self, nsamples, display_opencv=True):
        '''
        Show some random samples from the dataset
        '''
        if self.limit_masks:
            label_names = limit_multi_labels
        else:
            label_names = half_multi_task_labels

        sample_list = []
        dataloader = self.get_dataloader(nsamples, True, 0)
        batch = next(iter(dataloader))
        data_batch, target_batch = batch
        print("Displaying {} samples in {} mode. Dataset has {} items".format(nsamples, self.mode, len(self)))
        print("Data shape: {}".format(data_batch.shape))
        print("Target shape: {}".format(target_batch.shape))
        for batch_count, (inp, target) in enumerate(zip(data_batch, target_batch)):
            np_data = inp.numpy()
            np_target = target.numpy()

            data_display = np.zeros((np_data.shape[1], np_data.shape[2]*np_data.shape[0]), dtype=np_data.dtype)
            target_display = np.zeros((np_target.shape[1], np_target.shape[2]*np_target.shape[0]), dtype=np_target.dtype)

            for channel_count, channel in enumerate(np_data):
                data_display[0:np_data.shape[1],
                             np_data.shape[2]*channel_count:np_data.shape[2]*(channel_count + 1)
                             ] = channel
            for channel_count, channel in enumerate(np_target):
                target_display[0:np_target.shape[1],
                               np_target.shape[2]*channel_count:np_target.shape[2]*(channel_count + 1)
                               ] = imagePrint(channel, label_names[channel_count], org=(5, 5), scale=0.2)

            data_display = cv.resize(data_display, (0, 0), fx=2, fy=2)
            target_display = cv.resize(target_display, (0, 0), fx=2, fy=2)

            data_display = imagePrint(data_display, "{} sample {}".format(self.mode, batch_count), org=(10, 10), scale=0.5)

            sample_list.append((data_display, target_display))

            if display_opencv:
                cv.imshow("Input", data_display)
                cv.imshow("Target", target_display)
                if cv.waitKey(0) == 27:
                    quit()

        return sample_list


# Deprecated
class Cache():
    '''
    Global class holding pre-processed 3D data from multitask dataset
    Dynamic cache saving last used volumes, deleting old ones if not enough RAM
    Can load from the processed folder, and saves everything ever used there for use in subsequent runs.
    Assumes datas come "type compressed" (float16, uint8)
    Has adaptative memory limit
    '''
    EXPECTED_SIZE = 197
    MILLION = 1000000

    def __init__(self, savepath=multitask_hip_processed, decompress=True):
        self.cache_lock = Lock()
        self.decompress = decompress
        self.data = collections.OrderedDict()
        self.ram_size = psutil.virtual_memory().total / Cache.MILLION
        self.data_limit = 0.10*self.ram_size
        self.factor = 0.05*self.ram_size
        self.savepath = savepath
        self.first_limit_achieved = False
        self.loaded = False
        print("Initial cache memory limit: {}MB".format(int(self.data_limit)))

        os.makedirs(savepath, exist_ok=True)

        if self.decompress:
            print("Cache will return float32 data.")
        else:
            print("Cache will return data as stored.")

    def __getitem__(self, key):
        try:
            self.cache_lock.acquire()
            vol, mask = self.data[key]
            self.cache_lock.release()
        except KeyError:
            self.cache_lock.release()
            print("page miss, loading from HD")
            key_path = self.get_key_path(key)
            npz = np.load(key_path)
            vol, mask, orig = (npz['vol'], npz['mask'], npz['orig'])
            self.cache_lock.acquire()
            self.data[key] = (vol, mask, orig)
            self.cache_lock.release()
            self.check_size_limit_reached()

        if self.decompress:
            return (vol.astype(np.float32), mask.astype(np.float32), orig.astype(np.float32))
        else:
            return (vol, mask)

    def __setitem__(self, key, processed_data):
        '''
        Protected by locks
        '''
        vol, mask, orig = processed_data
        assert vol.dtype == np.dtype(np.float16) and mask.dtype == np.dtype(np.uint8) and orig.dtype == np.dtype(np.uint8), (
               " cache needs compressed data types!")

        self.cache_lock.acquire()
        self.data[key] = (vol, mask, orig)
        self.cache_lock.release()
        self.check_size_limit_reached()

        key_path = self.get_key_path(key)
        if not os.path.isfile(key_path):
            print("Key {} cache not found, saving to {}".format(key, key_path))
            np.savez_compressed(key_path, vol=vol, mask=mask, orig=orig)
        else:
            print("Key {} already in HD cache".format(key))

    def __len__(self):
        return len(self.data)

    def get_key_path(self, key):
        return add_path(self.savepath, key + '.npz')

    def check_size_limit_reached(self):
        '''
        Check and deletes oldest item if its exceeds intended size
        '''
        size = 0
        for k, v in self.data.items():
            size += v[0].nbytes + v[1].nbytes
        size = size/Cache.MILLION

        if self.first_limit_achieved:
            free = psutil.virtual_memory().available / Cache.MILLION
            if free < 0.2*self.ram_size:
                self.data_limit -= self.factor
                if self.data_limit <= 0:
                    self.data_limit = 0
                else:
                    print("Cache Memory limit decreased to {}MB due to high RAM usage".format((self.data_limit)))
            if free > 0.4*self.ram_size:
                self.data_limit += self.factor
                if self.data_limit >= 0.8*self.ram_size:
                    self.data_limit = 0.8*self.ram_size
                else:
                    print("Cache Memory limit increased to {}MB due to low RAM usage".format((self.data_limit)))

        if size >= self.data_limit:
            self.first_limit_achieved = True
            oldest_key = list(self.data.items())[:2]
            for ok in oldest_key:
                print("Deleted key {} cause of memory limits".format(ok[0]))
                del self.data[ok[0]]
            return True
        else:
            return False

    def load(self):
        load_deprecated = True
        if not self.loaded:
            print("Loading cache from HD to RAM...")
            if not load_deprecated:  # deprecated loading
                for f in glob.glob(add_path(self.savepath, '*.npz')):
                    npz = np.load(f)
                    self.cache_lock.acquire()
                    self.data[os.path.basename(os.path.basename(f).split('.')[0])] = (npz['vol'], npz['mask'])
                    self.cache_lock.release()
                    free = psutil.virtual_memory().available / Cache.MILLION
                    print("Free memory: {}".format(free), flush=True, end='\r')
                    if free < 0.5*self.ram_size:
                        print("Pre-load finished. Managed to pre-fill {} items".format(len(self)))
                        break
            self.loaded = True
        else:
            print("Cache already pre-loaded")


# Fixed mnihip ids division
fixed_mnihip_ids = {'test': ['42915', '42916', '42917', '42918', '42919', '42920', '42921', '42922', '42923', '42924', '42925',
                             '42926', '42927', '42928', '42929', '42930', '42931', '42933', '42934', '42912'],  # 42912
                    'train': ['31847', '31849', '31850', '31851', '31852', '31853', '31854', '31855', '31856', '32239',
                              '32241', '32242', '32243', '32244', '32245', '32246', '32247', '32248', '32249', '32252',
                              '32254', '32255', '32256', '32257', '32258', '32260', '32261', '32262', '32263', '32264',
                              '32265', '32266', '32268', '32269', '32270', '32271', '32272', '32273', '32274', '32275',
                              '32276', '32277', '32278', '32279', '32280', '32281', '32546', '32547', '32548', '32549',
                              '32550', '32551', '32552', '32553', '32554', '32555', '32556', '32557', '32558', '32559', '32560',
                              '32561', '32562', '32563', '32564', '32565', '32566', '32567', '32568', '32569', '32570', '32571',
                              '32572', '32573', '32574', '32575', '32576', '32577', '32578', '32579', '32580', '32581', '32582',
                              '32583', '33067', '33068', '33069', '33070', '33071', '33072', '33073', '33074', '33075',
                              '33076', '33077', '33078', '33079', '33080', '33081', '33082', '33083', '33084', '33085', '33086',
                              '33088', '33089', '33091', '33092', '33093', '33094', '33097', '33098',
                              '33099', '33100', '33101', '33103', '33104', '33105', '33107', '33758', '33760',
                              '33761', '33762', '33763', '33764', '33765', '33766', '33767', '33768', '33769', '33770', '33771',
                              '33772', '33773', '33774', '33775', '33777', '33778', '33779', '33780', '33781', '33782',
                              '33783', '33784', '33785', '33786', '33787', '33788', '33789', '33790', '33791', '33792',
                              '33794', '33795', '33796', '33797', '34423'],  # 34423 previously removed
                    'validation': ['34421', '34422', '42892', '42893', '42894', '42895', '42896', '42897', '42898',
                                   '42900', '42901', '42903', '42904', '42905', '42906', '42907', '42909', '42910',
                                   '42914', '42911']  # 42911 removed
                    }
fixed_mnihip_ids['all'] = fixed_mnihip_ids['train'] + fixed_mnihip_ids['validation'] + fixed_mnihip_ids['test']


def unit_test(image_dataset=True, dataset="harp", shuffle=False, ntoshow=1, show=True, plt_show=True, nworkers=0,
              hiponly=True, volume=False, e2d=False):
    '''
    Tests vizualisation of a training batch of the dataset
    '''
    # Long mask deprecated
    '''if adni or harp:
        long_mask = False
    else:
        long_mask = True'''
    from transforms import ReturnPatch, RandomFlip

    transform_list = [Compose([ReturnPatch(ppositive=1.0, patch_size=(64, 64), debug=True)]),
                      Compose([ReturnPatch(ppositive=0, patch_size=(64, 64), debug=True)]),
                      Compose([ReturnPatch(ppositive=1.0, patch_size=(64, 64), debug=True), RandomAffine(p=1.0, rotate=20,
                                                                                                         scale=(0.8, 1.2),
                                                                                                         debug=True)]),
                      Compose([ReturnPatch(ppositive=1.0, patch_size=(64, 64), debug=True), Intensity(p=1.0, brightness=0.1,
                                                                                                      force_max=True)]),
                      Compose([ReturnPatch(ppositive=1.0, patch_size=(64, 64), debug=True), Noisify(p=1.0)]),
                      Compose([ReturnPatch(ppositive=1.0, patch_size=(64, 64), debug=True), SoftTarget(p=1.0, order=10)]),
                      Compose([ReturnPatch(ppositive=1.0, patch_size=(64, 64), debug=True), RandomFlip(p=1.0,
                                                                                                       modes=['horflip'])])]

    print("Testing all orientations in all modes...")
    for train_transforms in transform_list:
        train_transforms.addto(ToTensor(debug=True), end=True)
        for m in ["test"]:
            for o in ["sagital"]:
                if dataset == "harp":
                    test = NewHARP(group="all", mode=m, orientation=o, fold=1, transform=train_transforms)
                elif dataset == "adni":
                    test = FloatHippocampusDataset(h5path=default_adni, mode=m, transform=train_transforms,
                                                   data_split=(0.5, 0.1, 0.4), adni=True, orientation=o, hiponly=True,
                                                   return_volume=False, e2d=True, mnireg=False)
                elif dataset == "oldharp":
                    test = FloatHippocampusDataset(h5path=default_harp, mode=m, harp=True, transform=train_transforms,
                                                   data_split=(0.7, 0.1, 0.2), adni=False, orientation=o, hiponly=True,
                                                   return_volume=False, e2d=True, mnireg=False, return_label=False)
                elif dataset == "mnihip":
                    # test = FloatHippocampusDataset(mode=m, transform=train_transforms, orientation=o, hiponly=hiponly,
                    #                                return_volume=volume, e2d=e2d)
                    test = MultiTaskDataset(verbose=True, hiponly=False, mode=m, transform=train_transforms, orientation=o,
                                            dim='2d', e2d=True, return_onehot=True, merge_left_right=True)
                else:
                    raise ValueError("Dataset {} does not exist".format(dataset))

                test_loader = data.DataLoader(test, batch_size=ntoshow, shuffle=shuffle, num_workers=0)
                batch = next(iter(test_loader))
                if show is True:
                    display_batch(batch, str(train_transforms) + o + " dataloader test in " + m)

        if plt_show:
            print("Showing " + str(train_transforms))
            plt.show()


def display_batch(batch, title):
    '''
    Displays a batch content on a grid
    '''
    if len(batch) == 2:
        imgs, tgts = batch
    elif len(batch) == 3:
        imgs, tgts, clss = batch

    print("display_batch input:", imgs.shape, tgts.shape)
    batch_len = len(imgs)
    grid_data = torch.zeros((batch_len, 1, imgs.size(2), imgs.size(3)))

    tgtsmax = tgts.max().float()
    if tgts.max() > 1:
        if not tgts.dtype == torch.long:
            raise ValueError("Tgts max higher than 1 and float?")
        tgts = tgts.float()/tgtsmax

    for i, (im, tg) in enumerate(zip(imgs, tgts)):
        if len(tg.shape) == 3 and tg.shape[0] != 1:
            buffer = torch.zeros((1, tg.shape[1], tg.shape[2]), dtype=tg.dtype)
            for j in range(1, tg.shape[0]):
                buffer += tg[j]
            buffer[buffer > 1] = 1.0
            buffer[buffer < 0] = 0.0
            tg = buffer

        if im.shape[0] == 3:
            overlap = im[1] + tg
        else:
            overlap = im + tg

        overlap[overlap > 1] = 1.0
        overlap[overlap < 0] = 0.0

        if len(batch) == 2:
            clas = None
        else:
            clas = HARP_CLASSES[clss[i].item()]

        overlap = torch.from_numpy(imagePrint(overlap.squeeze().numpy(), clas)).unsqueeze(0)

        grid_data[i] = overlap

    grid = torchvision.utils.make_grid(grid_data, nrow=batch_len//5).numpy().transpose(1, 2, 0)
    plt.figure(num=title)
    plt.title(str(batch_len) + " " + title + " samples")
    plt.axis('off')
    plt.imshow(grid)


def view_volumes(dataset_name="mnihip", wait=0, group=None, load_test=False, use_itk_snap=False, split=False):
    '''
    View volumes supplied by a dataset abstraction
    '''
    if dataset_name == "harp":
        fhd = NewHARP("all", mode="all", verbose=True)
    elif dataset_name == "mnihip":
        fhd = ConcatDataset((FloatHippocampusDataset(return_volume=True, transform=None, orientation="coronal", mode="train",
                                                     verbose=True),
                             FloatHippocampusDataset(return_volume=True, transform=None, orientation="coronal", mode="validation",
                                                     verbose=True),
                             FloatHippocampusDataset(return_volume=True, transform=None, orientation="coronal", mode="test",
                                                     verbose=True)))
    elif dataset_name == "oldharp":
        fhd = ConcatDataset((FloatHippocampusDataset(h5path=default_harp, return_volume=True, mode="train", adni=False, harp=True,
                                                     data_split=(0.7, 0.1, 0.2), mnireg=False, return_label=False),
                             FloatHippocampusDataset(h5path=default_harp, return_volume=True, mode="validation", adni=False,
                                                     harp=True, data_split=(0.7, 0.1, 0.2), mnireg=False, return_label=False),
                             FloatHippocampusDataset(h5path=default_harp, return_volume=True, mode="test", adni=False, harp=True,
                                                     data_split=(0.7, 0.1, 0.2), mnireg=False, return_label=False)))
    elif dataset_name == "mniharp":
        fhd = HARP("all", mode="all", mnireg=True)
    elif dataset_name == "cc359":
        fhd = CC359Data()
    elif dataset_name == "adni":
        fhd = FloatHippocampusDataset(h5path=default_adni, return_volume=True, mode="test", adni=True, data_split=(0.0, 0.0, 1.0),
                                      mnireg=False)
    elif dataset_name == "mniadni":
        fhd = FloatHippocampusDataset(h5path=mni_adni, return_volume=True, mode="test", adni=True, data_split=(0.0, 0.0, 1.0),
                                      mnireg=True)
    elif dataset_name == "multitask":
        fhd = MultiTaskDataset(verbose=False, hiponly=False, return_onehot=True)
    elif dataset_name == "hipmultitask":
        fhd = MultiTaskDataset(verbose=True, hiponly=True, return_onehot=False)
    else:
        raise ValueError("Dataset {} not recognized".format(dataset_name))

    itk_manager = ITKManager() if use_itk_snap else None
    print("Dataset size: {}".format(len(fhd)))
    for i in range(len(fhd)):
        label = None
        try:
            data = fhd[i]
            if len(data) == 2:
                im, ma = data
            elif len(data) == 3:
                im, ma, label = data

            # Handle 4D volumes (multichannel masks)
            if ma.ndim == 4:
                preserve_type = ma.dtype
                buffer = np.zeros(ma.shape[1:])
                for c in range(ma.shape[0]):
                    buffer = buffer + ma[c]*c
                buffer = buffer/buffer.max()
                ma = buffer.astype(preserve_type)
            else:
                ma = ma/ma.max()

            if not load_test:
                if hasattr(fhd, "multilabels"):
                    multilabel = fhd.multilabels
                else:
                    multilabel = None
                if split:
                    splitted_im = split_l_r(im)
                    splitted_ma = split_l_r(ma)
                    viewnii(splitted_im["left"], splitted_ma["left"], id=dataset_name + " left", wait=wait,
                            multi_labels=multilabel, label=label, itk_manager=itk_manager)
                    viewnii(splitted_im["right"], splitted_ma["right"], id=dataset_name + " right", wait=wait,
                            multi_labels=multilabel, label=label, itk_manager=itk_manager)
                else:
                    viewnii(im, ma, id=dataset_name, wait=wait, multi_labels=multilabel, label=label, itk_manager=itk_manager)
        except KeyboardInterrupt:
            print("Dataset test interrupted by Ctrl-C")
            quit()


class MultiTaskDataset(data.Dataset):
    '''
    Abstracts raw mnihip with all masks
    Should make FloatHippocampusDataset deprecated when completed
    Provides option to ramcache all data, for fast h5 file less slice extraction
    '''
    EXPECTED_SHAPE = (181, 217, 181)
    presence_dict = None
    cache = None
    @staticmethod
    def producer(r, data, queue):
        '''
        Gets data from dataset and puts in queue to go to cache
        '''
        for i in tqdm(r):
            queue.put(data[i])
        queue.put(None)

    @staticmethod
    def fill_hd_cache(nworkers=mp.cpu_count()):
        '''
        Fills ramcache with all volumes, ram cache is dict refering to ID, getitem will use correct ID
        RAM CACHE is class with get and set methods that interface float32 and float16
        '''
        assert nworkers >= 0, "nworkers cant be negative"
        data = MultiTaskDataset(verbose=True, return_id=True, use_raw_data=True, compress=True)
        if len(MultiTaskDataset.cache) != len(data):
            print("Cache file not found or incomplete, RAM caching being filled... This might take some time.")
            if nworkers == 0:
                for i in tqdm(range(len(data))):
                    Id, vol, mask, orig = data[i]
                    MultiTaskDataset.cache[Id] = (vol, mask, orig)
            else:
                print("Using multiprocessing with {} workers for RAM fill...".format(nworkers))
                queue = Queue(maxsize=(psutil.virtual_memory().total//Cache.MILLION)//8)
                data_len = len(data)
                for r in chunks(list(range(data_len)), data_len//nworkers):
                    ps = []
                    p = Process(target=MultiTaskDataset.producer, args=(r, data, queue))
                    ps.append(p)
                    p.start()

                done = 0
                print("Consuming data fetched by {} workers".format(nworkers))
                while True:
                    data = queue.get()
                    print("Get Got!")
                    if data is None:
                        done += 1
                    else:
                        vol_id, vol, mask, orig = data
                        MultiTaskDataset.cache[vol_id] = (vol, mask, orig)
                    if done == nworkers:
                        return

                for p in ps:
                    p.join()

        else:
            print("RAM cache already filled with {} entries".format(len(MultiTaskDataset.cache)))

    def __init__(self, path=multitask_path, group='both', mode='all', data_split=(0.8, 0.1, 0.1), transform=None, verbose=True,
                 hiponly=False, use_raw_data=False, dim='3d', orientation=None, e2d=False, return_id=False, compress=False,
                 return_onehot=True, zero_background=True, merge_left_right=True):
        '''
        path: path that contains folders ['controls', 'patients']
        group: select one of 'controls', 'patients', or 'both'
        mode: select one of 'all', 'train', 'validation', or 'test'
        data_split: how to separate data between modes, same separation not guaranteed if not using default value
        transform: transform to apply to the data, make sure its compatible with dim
        verbose: more prints if True
        hiponly: returns slices that dont have mask when dim == '2d' if True
        ramcache: stores all volumes on RAM in initialization phase if True
        dim: return slices if '2d', or volumes if '3d'
        orientation: when using dim='2d', selects orientation to return
        e2d: return input as 3 neighbour slices
        return_id: returns id, data, target instead of data, target
        compress: returns in a compressed datatype instead of float32
        return_onehot: returns target in onehot format
        zero_background: zero out onehot background (cross entropy should ignore index 0)
        merge_left_right: if true, do not return different label for left and right
        '''
        super(MultiTaskDataset, self).__init__()
        # Assert arguments make sense
        valid_modes = ['all', 'train', 'validation', 'test']
        valid_groups = ['both', 'controls', 'patients']
        valid_dims = ['2d', '3d']
        assert mode in valid_modes and group in valid_groups and dim in valid_dims, ("arguments to MultiTask dataset make no"
                                                                                     "sense check documentation")
        if dim == '3d':
            assert e2d is False and orientation is None, ("e2d and orientation makes no sense when dim == '3d', also hiponly has"
                                                          "to be False")
        assert hiponly != merge_left_right or (merge_left_right is False and hiponly is False), ("choose one: merge sides or"
                                                                                                 "hiponly")
        assert return_onehot, "option to not return one hot is currently disabled"
        self.transform = transform
        self.verbose = verbose
        self.hiponly = hiponly
        self.use_raw_data = use_raw_data
        self.e2d = e2d
        self.orientation = orientation
        self.return_onehot = return_onehot
        self.zero_background = zero_background
        self.merge_left_right = merge_left_right
        self.multilabels = HALF_MULTI_TASK_NCHANNELS if merge_left_right else MULTI_TASK_NCHANNELS

        if orientation is not None:
            self.n_slices = MultiTaskDataset.EXPECTED_SHAPE[orientations.index(orientation)]
            self.slice_mode = True
            # Initialize presence dict if not initialized already
            if MultiTaskDataset.presence_dict is None:
                with open(add_path(multitask_hip_processed, 'hip_presence.json'), 'r') as f:
                    MultiTaskDataset.presence_dict = json.load(f)

            self.presence_dict = copy.deepcopy(MultiTaskDataset.presence_dict)
            print("Presence dict initialized with {} volumes".format(len(self.presence_dict)))
        else:
            self.slice_mode = False
        self.dim = dim
        self.return_id = return_id
        self.compress = compress
        self.reconstruction_orientations = orientations

        if compress:
            self.vol_type, self.mask_type, self.orig_type = np.dtype(np.float16), np.dtype(np.uint8), np.dtype(np.uint8)
            print("WARNING: Returning compressed data")
        else:
            self.vol_type, self.mask_type, self.orig_type = np.dtype(np.float32), np.dtype(np.float32), np.dtype(np.float32)

        os_separator = "\\" if os.name == 'nt' else '/'
        files = list(glob.glob(add_path(path, "**", "*.nii"), recursive=True))

        if not self.slice_mode:
            MultiTaskDataset.cache = Cache()
            if len(glob.glob(add_path(MultiTaskDataset.cache.savepath, '*.npz'))) == Cache.EXPECTED_SIZE:
                MultiTaskDataset.cache.load()
                print("Datacache pre-loaded to limit, from {}".format(MultiTaskDataset.cache.savepath))

        # Store tuples dict['id'] = (vol, mask)
        controls = {}
        patients = {}

        # Get information from nii files
        for f in files:
            ftokens = f.split(os_separator)
            if not ('SOBROU' in ftokens):
                bname = os.path.basename(f)
                number = bname[-9:-4]
                if bname[0:3] == 'lab':
                    index = 1
                elif bname[0] == 'n':
                    index = 0
                else:
                    continue

                if "PACIENTES" in ftokens:
                    try:
                        patients[number]
                    except KeyError:
                        patients[number] = ['', '']

                    patients[number][index] = f
                elif "CONTROLES" in ftokens:
                    try:
                        controls[number]
                    except KeyError:
                        controls[number] = ['', '']

                    controls[number][index] = f

        # Remove ids not in selected mode
        for k in list(controls.keys()):
            if k not in fixed_mnihip_ids[mode]:
                controls.pop(k)
        for k in list(patients.keys()):
            if k not in fixed_mnihip_ids[mode]:
                patients.pop(k)

        citems = list(controls.items())
        pitems = list(patients.items())

        # Remove ids not in selected group
        if group == 'both':
            self.final_items = citems + pitems
            if self.verbose:
                print("{} control + patients items".format(len(self.final_items)))
        elif group == 'controls':
            self.final_items = citems
            if self.verbose:
                print("{} control items".format(len(self.final_items)))
        elif group == 'patients':
            self.final_items = pitems
            if self.verbose:
                print("{} patient items".format(len(self.final_items)))

        self.n_vols = len(self.final_items)

        if self.slice_mode:
            self.slice_paths = []

            for k, v in self.final_items:
                for slic in self.presence_dict[k][self.orientation]:
                    self.slice_paths.append(add_path(multitask_hip_processed_slices, k + '_' + self.orientation[0] + str(slic) +
                                                     ".npz"))

        print(("Multitask init in group {}, mode {}, datasplit {}, use_raw_data {}, compress {} return_onehot {}"
               "zero background {}, merge_left_right {}, done.").format(group, mode, data_split, use_raw_data, compress,
                                                                        return_onehot, zero_background, merge_left_right))

    def __len__(self):
        if self.slice_mode:
            return len(self.slice_paths)
        else:
            return self.n_vols

    def __getitem__(self, i):
        '''
        Masks are one hots
        '''
        gitem_time = time.time()

        if self.slice_mode:
            slice_path = self.slice_paths[i]
            if self.verbose:
                print(slice_path)
            mtask_id = slice_path.split('/')[-1].split('_')[0]
            self.slice_index = i
            if self.verbose:
                print("Slice came from {}, path {}".format(mtask_id, slice_path))
        else:
            mtask_id, (volpath, maskpath) = self.final_items[i]
            if self.verbose:
                print("Subject {} Volpath: {}".format(mtask_id, volpath))

        # Select slices if in slice_mode
        if self.slice_mode:
            npz = np.load(self.slice_paths[i])
            center_img = npz['vol_slice']
            target = npz['mask_slice'] if self.return_onehot else npz['orig_slice']

            if self.e2d:
                preindex = self.slice_index - 1
                if preindex < 0:
                    preindex = self.slice_index
                postindex = self.slice_index + 1
                if postindex == len(self):
                    postindex = self.slice_index

                image = np.zeros((3, center_img.shape[0], center_img.shape[1]), dtype=center_img.dtype)

                image[0] = np.load(self.slice_paths[preindex])['vol_slice']
                image[1] = center_img
                image[2] = np.load(self.slice_paths[postindex])['vol_slice']
            else:
                image, target = center_img, target

            image, target = image.astype(self.vol_type), target.astype(self.mask_type if self.return_onehot else self.orig_type)
        else:
            # Get the volume from memory
            if self.use_raw_data:
                # Gets raw data, process and returns
                vol = nib.load(volpath).get_fdata()
                mask = nib.load(maskpath).get_fdata()

                vs = vol.shape
                ms = mask.shape

                if vs != ms:
                    print("WARNING: shapes of mask and volume are not equal!")

                norm_vol = normalizeMri(vol.astype(np.float32)).astype(self.vol_type)
                norm_mask = int_to_onehot(mask, onehot_type=self.mask_type)
                norm_orig = mask.astype(self.orig_type)

            else:
                # get already processed data
                norm_vol, norm_mask, norm_orig = MultiTaskDataset.cache[mtask_id]

            image = norm_vol
            if self.return_onehot:
                target = norm_mask
            else:
                target = norm_orig

        target_type = self.mask_type if self.return_onehot else self.orig_type

        assert image.dtype == self.vol_type and target.dtype == target_type, (" wrong vol or mask dtype before transforms,"
                                                                              "should be {} and {}").format(self.vol_type,
                                                                                                            self.mask_type)

        if self.hiponly:
            if self.return_onehot:
                target = target[11] + target[12]
            else:
                target = np.ma.masked_not_equal(target, 11).filled(0) + np.ma.masked_not_equal(target, 12).filled(0)
        elif self.merge_left_right:
            if self.return_onehot:
                new_target = np.zeros((HALF_MULTI_TASK_NCHANNELS,) + target.shape[1:], dtype=target.dtype)
                new_target[0] = target[0]
                for i in range(1, HALF_MULTI_TASK_NCHANNELS):
                    new_target[i] = target[2*i-1] + target[2*i]
            else:
                new_target = np.zeros_like(target)
                new_target = (target + 1)//2
            target = new_target

        # Remove background if returning one hot. If not returning one hot, make sure ignore_index 0 is on cross entropy.
        if self.return_onehot and self.zero_background:
            if target.ndim == 3:
                target[0, :, :] = 0
            elif target.ndim == 4:
                target[0, :, :, :] = 0
            else:
                raise ValueError("Strange ndim in target")
            assert target[0].max() == 0

        if self.transform is not None:
            image, target = self.transform(image, target)

        if self.verbose:
            print("Time that took to get [{}, {}] {} {}s".format(image.shape, target.shape,
                                                                 "slice" if self.slice_mode else "volume",
                                                                 time.time() - gitem_time))

        if self.return_id:
            return mtask_id, image, target, norm_orig
        else:
            return image, target


def get_harp_group():
    if '-ad' in argv:
        group = "ad"
    elif '-cn' in argv:
        group = "cn"
    elif '-mci' in argv:
        group = "mci"
    else:
        group = "all"

    print("Selected HarP group: {} group".format(group))

    return group


class NewHARP(data.Dataset):
    '''
    New standalone HARP class that handles correct slice and volume selection.
    Pre-processed volumes and slices are supposed to be saved in processed and processed_slices in the deafult_harp path
    List of all volume IDs per classification class inside class.
    Since order of list in fixed, fold can be selected by slicing.
    '''
    KEY_LIST = {'cn': ['029_S_4279', '009_S_0862', '032_S_0677', '094_S_4460', '029_S_4385', '133_S_0525', '020_S_1288',
                       '010_S_0067', '003_S_0931', '002_S_1280', '009_S_0842', '011_S_0021', '073_S_0089', '002_S_0559',
                       '127_S_0259', '023_S_0061', '005_S_0602', '007_S_4387', '002_S_0685', '018_S_0425', '011_S_0002',
                       '002_S_0295', '006_S_4449', '037_S_0303', '002_S_0413', '100_S_0047', '011_S_0005', '129_S_4369',
                       '003_S_0907', '100_S_0015', '127_S_0260', '123_S_0072', '002_S_1261', '123_S_0106', '002_S_4225',
                       '100_S_1286', '098_S_0172', '023_S_0031', '010_S_0419', '013_S_4731', '016_S_4121', '032_S_0479',
                       '013_S_1276', '011_S_0016'],
                'mci': ['123_S_0108', '013_S_0325', '136_S_0579', '002_S_1070', '003_S_0908', '002_S_0782', '136_S_0429',
                        '006_S_0322', '003_S_1074', '011_S_0241', '023_S_0030', '005_S_0448', '002_S_0729', '009_S_1030',
                        '123_S_0050', '131_S_0384', '003_S_1122', '013_S_1186', '127_S_0397', '023_S_0331', '007_S_0128',
                        '012_S_1292', '012_S_1321', '002_S_0954', '016_S_1092', '023_S_0604', '023_S_1247', '011_S_0856',
                        '010_S_0422', '100_S_0892', '123_S_0390', '094_S_1293', '016_S_0769', '005_S_0222', '131_S_1389',
                        '123_S_1300', '100_S_0995', '016_S_1138', '126_S_1340', '002_S_1155', '003_S_1057', '023_S_0376',
                        '127_S_0393', '100_S_0006', '029_S_1073', '007_S_0101'],
                'ad': ['067_S_1185', '003_S_4136', '123_S_0162', '009_S_1334', '003_S_4142', '005_S_1341', '006_S_4192',
                       '126_S_0784', '018_S_4696', '067_S_0812', '100_S_1062', '130_S_0956', '003_S_1059', '123_S_0091',
                       '023_S_0916', '013_S_0592', '016_S_0991', '127_S_0844', '031_S_1209', '023_S_0139', '020_S_0213',
                       '011_S_0010', '027_S_1081', '002_S_0938', '005_S_0221', '002_S_0816', '016_S_1263', '067_S_1253',
                       '023_S_1289', '127_S_0754', '098_S_0149', '011_S_0183', '123_S_0088', '019_S_4549', '027_S_1385',
                       '003_S_1257', '131_S_0691', '012_S_0689', '094_S_4089', '123_S_0094', '019_S_4252', '126_S_0606',
                       '130_S_4730', '082_S_1079', '007_S_1304']}

    def __init__(self, group="all", mode="all", fold=None, transform=None, verbose=True, return_details=False,
                 return_onehot=False, orientation=None, e2d=True):
        '''
        Based on KEY_LIST splits files accordingly in train/val and test. Should be consistent with slices and volumes. Specially
        not mixing slices from the same individuals in different folds.
        '''
        super(NewHARP, self).__init__()
        if mode != "all":
            assert fold in range(1, 6)
        assert mode in ["all", "train", "validation", "test"]
        assert group in ["all"] + HARP_CLASSES
        assert orientation in [None, "sagital", "coronal", "axial"]
        type_assert(bool, verbose, return_details, return_onehot, e2d, return_onehot)
        self.reconstruction_orientations = orientations
        self.fold = fold
        self.mode = mode
        self.group = HARP_CLASSES if group == "all" else [group]
        self.return_details = return_details
        self.return_onehot = return_onehot
        self.transform = transform
        self.keys = []
        self.verbose = verbose
        self.orientation = orientation
        self.e2d = e2d
        for g in self.group:
            self.keys = self.keys + NewHARP.KEY_LIST[g]

        full_len = len(self.keys)
        train_len, test_len = int(full_len*0.7), int(full_len*0.2)

        # Fold Selection
        if fold in range(1, 6):
            self.train_val_set = self.keys[:(fold-1)*test_len] + self.keys[fold*test_len:]
            self.test_set = self.keys[(fold-1)*test_len:fold*test_len]

            self.train_set = self.train_val_set[:train_len]
            self.val_set = self.train_val_set[train_len:]

        if mode == "train":
            self.keys = self.train_set
        elif mode == "validation":
            self.keys = self.val_set
        elif mode == "test":
            self.keys = self.test_set

        if verbose:
            print("IDs used:")
            print(self.keys)

        # Get slices from folders in keys
        if self.orientation is not None:
            self.slices = []
            for key in self.keys:
                self.slices = self.slices + glob.glob(add_path(default_harp, "processed_slices", key,
                                                      self.orientation[0] + "*.npz"))
            self.keys = self.slices

        final_len = len(self.keys)

        print("HARP args -> group: {}, mode: {}, fold: {}, transform: {}, verbose: {}, return_details: {}, "
              "return_onehot: {}, volumetric lenght: {}".format(group, mode, fold, transform, verbose, return_details,
                                                                return_onehot, final_len))

    def __len__(self):
        '''
        Returns how many volumes/slices are in the dataset
        '''
        return len(self.keys)

    def __getitem__(self, i):
        '''
        Returns slice/volume according to parameters set in init
        '''
        key = self.keys[i]
        if self.verbose:
            print("Subject: {}".format(key))

        if self.orientation is None:
            npz = np.load(add_path(default_harp, "processed", key + ".npz"))
            data = npz["vol"]
        else:
            # in slice mode, keys are already full paths
            npz = np.load(key)
            if self.e2d:
                center_data = npz["vol"]
                slice_number = int(os.path.basename(key).split('.')[0][1:])
                post_slice = add_path(os.path.dirname(key), self.orientation[0] + str(slice_number + 1) + ".npz")
                pre_slice = add_path(os.path.dirname(key), self.orientation[0] + str(slice_number - 1) + ".npz")

                if os.path.exists(post_slice):
                    post_data = np.load(post_slice)["vol"]
                else:
                    post_data = center_data

                if os.path.exists(pre_slice):
                    pre_data = np.load(pre_slice)["vol"]
                else:
                    pre_data = center_data

                data = np.zeros((3, center_data.shape[0], center_data.shape[1]), dtype=center_data.dtype)
                pre_shape, center_shape, post_shape = pre_data.shape, center_data.shape, post_data.shape

                if (pre_shape != post_shape or pre_shape != center_shape or center_shape != post_shape):
                    if self.verbose:
                        print("Concatenating slices of different shapes")
                    cc = CenterCrop(*center_shape)
                    pre_data, post_data = cc(pre_data), cc(post_data)

                data[0] = pre_data
                data[1] = center_data
                data[2] = post_data
            else:
                data = npz["vol"]

        if self.return_onehot:
            target = npz["onehot_target"]
            assert one_hot(torch.from_numpy(target).unsqueeze(0))
        else:
            target = npz["class_target"]

        if self.transform is not None:
            data, target = self.transform(data, target)

        if self.return_details:
            label = None

            for k, v in NewHARP.KEY_LIST.items():
                if key in v:
                    label = k

            if label is None:
                raise RuntimeError("Label for key {} was not found...?".format(k))

            return data, target, label, key
        else:
            return data, target

    @staticmethod
    def process_rawvolumes():
        '''
        Populates processed folder
        '''
        for k, v in tqdm(NewHARP.KEY_LIST.items()):
            for harp_id in tqdm(v):
                vol_path = add_path(default_harp, "samples", k, harp_id + ".nii.gz")
                mask_path = add_path(default_harp, "masks", "all", harp_id + ".nii.gz")
                vol = nib.load(vol_path).get_fdata()
                hip = nib.load(mask_path).get_fdata()
                norm_vol = normalizeMri(vol.astype(np.float32))
                norm_hip = hip.astype(np.bool).astype(np.float32)
                onehot_hip = int_to_onehot(norm_hip)
                assert onehot_hip.shape[0] == 2
                np.savez_compressed(add_path(default_harp, "processed", harp_id), vol=norm_vol, class_target=norm_hip,
                                    onehot_target=onehot_hip)

    @staticmethod
    def process_slices():
        '''
        Populates processed slices
        '''
        db = NewHARP()
        onehot_db = NewHARP(return_onehot=True, return_details=True)

        for i in tqdm(range(len(db))):
            data, class_target = db[i]
            _, onehot_target, label, key = onehot_db[i]
            os.makedirs(add_path(default_harp, "processed_slices", key), exist_ok=True)
            print(data.shape, class_target.shape, onehot_target.shape)
            for orientation in tqdm(range(3)):
                for slice_index in tqdm(range(data.shape[orientation])):
                    class_slice = get_slice(class_target, slice_index, orientations[orientation], rotate=90)
                    if class_slice.sum() > 0:
                        _slice = get_slice(data, slice_index, orientations[orientation], rotate=90)
                        onehot_slice = get_slice(onehot_target, slice_index, orientations[orientation], rotate=90)
                        np.savez_compressed(add_path(default_harp, "processed_slices", key,
                                                     orientations[orientation][0] + str(slice_index)) + ".npz", vol=_slice,
                                            class_target=class_slice, onehot_target=onehot_slice)


# TODO Marked for deprecation
class HARP(data.Dataset):
    '''
    Abstracts volumetric HARP data
    '''
    harp_classes_dict = {}
    @staticmethod
    def check_id_class(harpid):
        assert len(harpid.split('_')) == 3, "harpid {} not valid"
        for i, cl in enumerate(HARP_CLASSES):
            if harpid in HARP.harp_classes_dict[cl]:
                return i
        raise ValueError("{} does not have a harp class. You probably gave a wrong harpid".format(harpid))

    def __init__(self, group, mode='all', data_split=(0.7, 0.1, 0.2), transform=None, verbose=True, mnireg=False,
                 return_label=False, return_harp_id=False):
        '''
        Group should be one of 'ad', 'cn', 'mci'
        '''
        super(HARP, self).__init__()
        assert group in ['ad', 'cn', 'mci', 'all'], " harp group should be one of 'ad', 'cn', 'mci', is {}".format(group)
        assert mode in ['train', 'validation', 'test', 'all'], " mode {} not supported by harp".format(mode)
        if mode == 'all':
            data_split = (0, 0, 1)
            mode = 'test'
            if verbose:
                print("HARP using all data of the {} group".format(group))
        else:
            if verbose:
                print("HARP initializing in {} mode with data split {}, using the {} group"
                      "return_label {} and return_harp_id {}".format(mode, data_split, group, return_label, return_harp_id))
        assert np.sum(np.array(data_split)) == 1.0, "data_split should sum to 1.0"
        self.mnireg = mnireg
        self.transform = transform
        if mnireg:
            self.harp_path = mni_harp
            self.volume_shape = (182, 218, 182)  # only used for mnihip data
        else:
            self.harp_path = default_harp
        self.group = group
        self.samples = []
        self.return_label = return_label
        self.return_harp_id = return_harp_id
        if group == 'all':
            for g in ['ad', 'cn', 'mci']:
                self.samples += glob.glob(add_path(self.harp_path, "samples", g, "*.nii.gz"))
        else:
            self.samples = glob.glob(add_path(self.harp_path, "samples", self.group, "*.nii.gz"))
        self.reconstruction_orientations = orientations
        self.verbose = verbose
        if len(self.samples) == 0:
            raise ValueError("nii data not found, check if you have data in {}".format(self.harp_path))

        train_split = int(data_split[0]*len(self.samples))
        val_split = train_split + int(data_split[1]*len(self.samples))

        if mode == "train":
            self.samples = self.samples[:train_split]
        elif mode == "validation":
            self.samples = self.samples[train_split:val_split]
        elif mode == "test":
            self.samples = self.samples[val_split:]

    def __len__(self):
        return len(self.samples)

    def get_by_name(self, name):
        '''
        Name should be only harp_id without format (without .nii or .nii.gz)
        Returns none if name is not found
        '''
        for s in self.samples:
            if os.path.basename(s).split(".nii.gz")[0] == name:
                return self.__getitem__(self.samples.index(s))
        return None

    def __getitem__(self, i):
        sample = self.samples[i]

        fname = os.path.basename(sample)
        harp_id = fname.split('.nii.gz')[0]

        class_label = HARP.check_id_class(harp_id)

        volpath = sample
        vol = nib.load(volpath).get_fdata()
        if self.group == 'all':
            hip = nib.load(add_path(self.harp_path, "masks", self.group, harp_id + ".nii.gz")).get_fdata()
        else:
            l_hippath = add_path(self.harp_path, "masks", self.group, harp_id + "_L.nii.gz")
            r_hippath = add_path(self.harp_path, "masks", self.group, harp_id + "_R.nii.gz")

            hip = nib.load(l_hippath).get_fdata() + nib.load(r_hippath).get_fdata()
        if self.verbose:
            print("Subject {} Volpath: {}".format(harp_id, volpath))
        vs = vol.shape
        hs = hip.shape

        if vs != hs:
            print("WARNING: shapes of mask and volume are not equal!")

        norm_vol = normalizeMri(vol.astype(np.float32))
        norm_hip = hip.astype(np.bool).astype(np.float32)

        if self.transform is not None:
            norm_vol, norm_hip = self.transform(norm_vol, norm_hip)

        if self.return_harp_id:
            assert self.return_label
            return norm_vol, norm_hip, class_label, harp_id
        elif self.return_label:
            return norm_vol, norm_hip, class_label
        else:
            return norm_vol, norm_hip


try:
    with open(os.path.join(default_harp, "harp_classes.pkl"), 'rb') as fl:
        HARP.harp_classes_dict = pickle.load(fl)
except FileNotFoundError:
    try:
        with open("harp_classes.pkl") as fl:
            HARP.harp_classes_dict = pickle.load(fl)
    except FileNotFoundError:
        print("WARNING: Could not find harp classes file. HARP dataset will not work correctly.")  # development error


class ADNI(data.Dataset):
    '''
    Abstracts ADNI data
    Subjects ADNI: 002_S_0295, 002_S_0413, 002_S_0619, 002_S_0685, 002_S_0729, 002_S_0782, 002_S_0938, 002_S_0954, 002_S_0954,
                   002_S_1018, 002_S_1155, 003_S_0907, 003_S_0981, 005_S_0222, 005_S_0324, 005_S_0448, 005_S_0546, 005_S_0553,
                   005_S_0572, 005_S_0610, 005_S_0814, 005_S_1341, 006_S_0498, 006_S_0675, 006_S_0681, 006_S_0731, 006_S_1130,
                   007_S_0041, 007_S_0068, 007_S_0070, 007_S_0101, 007_S_0249, 007_S_0293, 007_S_0316, 007_S_0414, 007_S_0698,
                   007_S_1206, 007_S_1222, 007_S_1339, 009_S_0751, 009_S_1030, 011_S_0003, 011_S_0005, 011_S_0010, 011_S_0016,
                   011_S_0021, 011_S_0022, 011_S_0023, 011_S_0053, 011_S_0168
    '''
    def __init__(self, mnireg=False):
        super(ADNI, self).__init__()
        if mnireg:
            print("Using MNI registered ADNI!")
            self.adnipath = mni_adni
        else:
            print("Using isometric ADNI!")
            self.adnipath = default_adni
        self.samples = glob.glob(self.adnipath + "/samples/*.nii.gz")
        self.reconstruction_orientations = orientations
        if len(self.samples) == 0:
            raise ValueError("nii data not found, check if you have data in {}".format(self.adnipath))

    def __len__(self):
        return len(self.samples)

    def get_by_name(self, name):
        '''
        Name should be only id without format (without .nii)
        '''
        return self.__getitem__(self.samples.index(add_path(self.adnipath, "samples", name + ".nii.gz")))

    def __getitem__(self, i):
        sample = self.samples[i]
        fname = os.path.basename(sample)

        volpath = sample
        hippath = add_path(self.adnipath, "masks", fname)

        print("Subject {} Volpath: {}".format(fname.split('.')[0], volpath))
        vol = nib.load(volpath).get_fdata()
        hip = nib.load(hippath).get_fdata()
        vs = vol.shape
        hs = hip.shape

        if vs != hs:
            print("WARNING: shapes of mask and volume are not equal!")

        norm_vol = normalizeMri(vol.astype(np.float32))
        norm_hip = hip.astype(np.bool).astype(np.float32)

        return norm_vol, norm_hip


class FloatHippocampusDataset(data.Dataset):
    '''
    Initializes dataset class abstracting slices structure
    h5path: path to h5 file containing all the data, in samples and masks groups containing a dataset for each volume
    orientation: one of sagital, coronal or axial
    mode: one of train, validation or test
    data_split: distribution of data over modes (train, val, test)
    transform: transforms to apply on slices. transforms should support 2 numpy arrays as input
    hiponly: only return slices with hippocampus presence
    return_volume: return volumes instead of slices
    '''
    def __init__(self, h5path=default_datapath, adni=False, harp=False, orientation="coronal", mode="train",
                 data_split=(0.8, 0.1, 0.1), transform=None, hiponly=False, return_volume=False, float16=True, e2d=False,
                 verbose=True, mnireg=True, return_label=False):
        super(FloatHippocampusDataset, self).__init__()
        assert adni != harp or (adni is False and harp is False), " cant be adni and harp at the same time"
        if adni:
            self.db = 'adni'
        elif harp:
            # TODO Marked for deprecation
            self.db = 'harp'
        else:
            self.db = 'mnihip'
        orientations = ["sagital", "coronal", "axial"]
        self.reconstruction_orientations = orientations
        modes = ["train", "validation", "test"]
        # self.deleted_vols = ["42911", "42912", "42913", "34423"]  # volumes removed for some reason
        self.deleted_vols = []
        assert orientation in orientations, "orientation should be one of {}".format(orientations)
        assert mode in modes, "mode should be one of {}".format(modes)
        assert np.sum(np.array(data_split)) == 1.0, "data_split should sum to 1.0"
        if return_volume:
            assert orientation == "coronal" and hiponly is False, ("orientation should be coronal and hiponly false when"
                                                                   "returning volumes")
        self.file_lock = adni_lock if adni else cla_lock
        self.float16 = float16
        self.tofloat32 = ToFloat32()
        self.h5path = h5path
        self.orientation = orientation
        self.transform = transform
        self.return_volume = return_volume
        self.e2d = e2d
        self.adni = adni
        self.harp = harp
        self.verbose = verbose
        self.multilabels = None
        self.return_label = return_label
        print(("FHD slices {} dataset initialized using {} data, returning volumes? {}"
              "MNI registered? {}").format(mode, self.db, self.return_volume, mnireg))

        self.volume_shape = (181, 217, 181)  # only used for mnihip data
        if adni:
            assert return_label is False, "adni does not have label support yet"
            self.vols = ADNI(mnireg=mnireg)
            self.fname = os.path.join(h5path,
                                      "float16_mniadni_hip_data_hiponly.hdf5" if mnireg else "float16_adni_hip_data_hiponly.hdf5")
        elif harp:
            assert not mnireg, "harp is not mniregistered, check arguments"
            self.vols = HARP(group='all', verbose=True, return_label=return_label)
            self.fname = os.path.join(h5path, "float16_harp_hip_data_hiponly.hdf5")
        elif hiponly:
            assert mnireg, "mnihip is always mni registered"
            assert return_label is False, "mnihip does not have label support yet"
            self.fname = os.path.join(h5path, "mni_hip_data_hiponly.hdf5")
        else:
            assert mnireg, "mnihip is always mni registered"
            assert return_label is False, "mnihip does not have label support yet"
            self.fname = os.path.join(h5path, "mni_hip_data_full.hdf5")

        self.file_lock.acquire()
        with h5py.File(self.fname) as h5file:
            try:
                samples = h5file["samples"][orientation]
            except KeyError:
                print("Problem with H5PY file {}, does it exist?".format(self.fname))
                quit()

            self.ids = list(samples.keys())
        self.file_lock.release()

        self.separator = '-' if harp else '_'

        # Remove slices of deleted volumes
        it = iter(copy.deepcopy(self.ids))
        if not adni:
            for k, i in enumerate(it):
                vid = i.split(self.separator)[0]
                if vid in self.deleted_vols:
                    self.ids.remove(i)
        if data_split != (0.8, 0.1, 0.1) or self.db != "mnihip":
            print("Initializing IDs with OLD method.")
            train_split = int(data_split[0]*len(self.ids))
            val_split = train_split + int(data_split[1]*len(self.ids))

            if mode == "train":
                self.ids = self.ids[:train_split]
            elif mode == "validation":
                self.ids = self.ids[train_split:val_split]
            elif mode == "test":
                self.ids = self.ids[val_split:]

            self.volume_ids = []
            for n, i in enumerate(self.ids):
                vid = i.split(self.separator)[0]
                if vid not in self.volume_ids:
                    self.volume_ids.append(vid)
        else:
            print("Initializing IDs with new method.")
            self.volume_ids = fixed_mnihip_ids[mode]
            for i in self.ids:
                if i.split(self.separator)[0] not in self.volume_ids:
                    self.ids.remove(i)

        print(mode + " detected volumes in: " + str(self.volume_ids))

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
            if self.return_volume:  # return volume
                samples = h5file["samples"]["coronal"]
                masks = h5file["masks"]["coronal"]
                vi = self.volume_ids[index]
                if self.adni or self.harp:
                    if vi == "16":
                        vi = "10"
                    data = self.vols.get_by_name(vi)
                    if len(data) == 2:
                        image, target = data
                    elif len(data) == 3:
                        image, target, label = data
                else:
                    image = np.zeros(self.volume_shape, dtype=np.float32)
                    target = np.zeros(self.volume_shape, dtype=np.float32)
                    j = 0
                    for i in self.ids:
                        if vid is not None:
                            vi = vid
                        if i.split(self.separator)[0] == vi:
                            # Rotate back to original volume orientation
                            image[:, j, :], target[:, j, :] = myrotate(samples.get(i)[:], -90), myrotate(masks.get(i)[:], -90)
                            j += 1
                    if self.verbose:
                        print("Subject " + str(vi))

            else:  # return slice
                samples = h5file["samples"][self.orientation]
                masks = h5file["masks"][self.orientation]
                i = self.ids[index]

                if self.harp and self.return_label:
                    label = HARP.check_id_class(i.split('-')[0])

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

                    vol_i = i.split(self.separator)[0]
                    if pri.split(self.separator)[0] != vol_i:
                        pri = i
                    if poi.split(self.separator)[0] != vol_i:
                        poi = i

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

        if self.return_label:
            return image, (target, label)
        else:
            return image, target


class CC359Data(data.Dataset):
    '''
    Abstracts public CC359 (https://sites.google.com/view/calgary-campinas-dataset/home)
    '''
    def __init__(self, cc359path="/home/diedre/bigdata/CC-359/Original",
                 maskpath="/home/diedre/bigdata/CC-359/Hippocampus-segmentation/Automatic/hippodeep"):
        super(CC359Data, self).__init__()
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


def get_adni_3d_dataloader(data_transforms, nworkers=4, batch_size=1, data_split=(0, 0, 1), mnireg=True):
    db = FloatHippocampusDataset(h5path=mni_adni if mnireg else default_adni, transform=data_transforms, return_volume=True,
                                 mode="test", adni=True, data_split=data_split, mnireg=mnireg)
    print("ADNI volumetric dataloader size: {} MNI? {}".format(len(db), mnireg))
    return data.DataLoader(db, batch_size=batch_size, shuffle=True, num_workers=nworkers), len(db)


def get_data(data_transforms, db="mnihip", datapath=default_datapath, nworkers=4, e2d=False, batch_size=20, test=False,
             volumetric=False, hiponly=True, return_volume=False, shuffle={"train": True, "validation": False, "test": False},
             nlr=False, ce=False, classify=False, gdl=False, fold=None):
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
            hips[o] = {m: FloatHippocampusDataset(h5path=default_adni, adni=True, data_split=(0.5, 0.1, 0.4), mode=m,
                                                  orientation=o, transform=data_transforms[m], hiponly=True, return_volume=False,
                                                  e2d=e2d) for m in modes}
    elif db == "concat":
        print("Concatenating adni and mnihip")
        for o in orientations:
            hips[o] = {m: data.ConcatDataset([FloatHippocampusDataset(h5path=mni_adni, adni=True, data_split=(0.5, 0.1, 0.4),
                                              mode=m,  orientation=o, transform=data_transforms[m], hiponly=True,
                                              return_volume=False, e2d=e2d, mnireg=True),
                                              FloatHippocampusDataset(h5path=datapath, mode=m, orientation=o,
                                                                      transform=data_transforms[m], hiponly=True,
                                                                      return_volume=False, e2d=e2d)]) for m in modes}
    elif db == "oldharp":
        print("Getting dataloaders for old harp slice data")
        for o in orientations:
            hips[o] = {m: FloatHippocampusDataset(h5path=default_harp, harp=True, adni=False, data_split=(0.7, 0.1, 0.2), mode=m,
                                                  orientation=o, transform=data_transforms[m], hiponly=True, return_volume=False,
                                                  e2d=e2d, mnireg=False, return_label=classify) for m in modes}
    elif db == "harp":
        print("Getting dataloaders for harp slice data")
        for o in orientations:
            hips[o] = {m: NewHARP(mode=m, fold=fold, transform=data_transforms[m],
                                  orientation=o, return_onehot=ce, e2d=e2d, verbose=False) for m in modes}
    elif db == "mixharp":
        print("Concatenating HARP and mnihip")
        for o in orientations:
            hips[o] = {m: data.ConcatDataset([FloatHippocampusDataset(h5path=default_harp, mnireg=False, harp=True, adni=False,
                                                                      data_split=(0.7, 0.1, 0.2), mode=m, orientation=o,
                                                                      transform=data_transforms[m], hiponly=True,
                                                                      return_volume=False, e2d=e2d, return_label=classify),
                                              FloatHippocampusDataset(h5path=datapath, mode=m, orientation=o,
                                                                      transform=data_transforms[m], hiponly=True,
                                                                      return_volume=False, e2d=e2d,
                                                                      return_label=classify)]) for m in modes}
    elif db == "mnihip":
        print("Using mnihip")
        for o in orientations:
            hips[o] = {m: FloatHippocampusDataset(h5path=datapath, mode=m, orientation=o, transform=data_transforms[m],
                                                  hiponly=hiponly, return_volume=return_volume, e2d=e2d) for m in modes}
    elif db == "multitask":
        print("Using multitask dataset")
        for o in orientations:
            hips[o] = {m: MultiTaskDataset(mode=m, orientation=o, group='both', verbose=False, dim='2d',
                                           transform=data_transforms[m], hiponly=False, e2d=e2d,
                                           return_onehot=True, zero_background=not(ce or gdl),
                                           merge_left_right=nlr) for m in modes}
    elif volumetric:
        if db == "mnihip3d":
            hips = {m: FloatHippocampusDataset(h5path=datapath, orientation="coronal", hiponly=False, mode=m,
                                               transform=data_transforms[m], return_volume=True, verbose=False) for m in modes}
        elif db == "harp3d":
            hips = {m: FloatHippocampusDataset(h5path=default_harp, harp=True, adni=False, data_split=(0.7, 0.1, 0.2),
                                               orientation="coronal", hiponly=False, mode=m, transform=data_transforms[m],
                                               return_volume=True, verbose=False, mnireg=False) for m in modes}
        elif db == "harp3dfold":
            hips = {m: NewHARP(fold=fold, mode=m, transform=data_transforms[m], orientation=None,
                               return_onehot=False, verbose=False, e2d=e2d) for m in modes}
        elif db == "nathip":
            group = get_group()
            hips = {m: NatHIP(fold=fold, mode="all", transform=data_transforms["test"], orientation=None,
                              return_onehot=False, verbose=False, e2d=False, group=group) for m in modes}
        else:
            raise ValueError("db {} not supported in volumetric mode".format(db))
    else:
        raise ValueError("db {} not supported".format(db))

    hip_dataloaders = {}

    if volumetric:
        bs = {m: batch_size//int((batch_size - 1)*(m == 'test') + 1) for m in modes}
    else:
        bs = {m: batch_size//int(19*(m == 'test') + 1) for m in modes}
    print("batch sizes: {}".format(bs))

    for m in modes:
        if bs[m] < 1:
            bs[m] = 1
    if volumetric:
        hip_dataloaders = {m: data.DataLoader(hips[m], batch_size=bs[m], shuffle=shuffle[m], num_workers=nworkers) for m in modes}
    else:
        for o in orientations:
            hip_dataloaders[o] = {m: data.DataLoader(hips[o][m], batch_size=bs[m], shuffle=shuffle[m],
                                                     num_workers=nworkers) for m in modes}

    print("dataset_sizes:")
    if volumetric:
        dataset_sizes = {m: len(hips[m]) for m in modes}
    else:
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


def show_mean_mask(db, name, dic, shape, shared_array, hist_build):
    try:
        db_vol_mean, db_mean = dic[name]
    except KeyError:
        if shape is not None:
            db_vol_mean = np.zeros(shape, dtype=np.float32)
            db_mean = np.zeros(shape, dtype=np.float32)

        for i in tqdm(range(len(db))):
            vol, mask = db[i]

            if hist_build:
                vol = (vol*255).astype(np.uint8)
                hist = np.histogram(vol, bins=range(256))[0]
                for pos, value in enumerate(hist):
                    shared_array[pos] += value

            if shape is not None:
                db_vol_mean += vol
                db_mean += mask

        if shape is not None:
            db_vol_mean = db_vol_mean / len(db)
            db_mean = db_mean / len(db)

            nib.save(nib.Nifti1Image(db_vol_mean, affine=None), os.path.join("data", "{}_mean_vol.nii.gz".format(name)))
            nib.save(nib.Nifti1Image(db_mean, affine=None), os.path.join("data", "{}_mean_mask.nii.gz".format(name)))

            dic[name] = [db_vol_mean, db_mean]


def compare_annotations(dbs, names, shape, hist_build=False):
    '''
    Compare annotations from datasets with stable volume size (registered or center croped)
    '''
    m = Manager()
    dic = m.dict()
    ps = []
    shared_arrays = {name: mp.Array('i', [0 for _ in range(256)]) for name in names}

    if not hist_build:
        save_names = [(os.path.join("data", "{}_mean_vol.nii.gz".format(names[i])),
                       os.path.join("data", "{}_mean_mask.nii.gz".format(names[i]))) for i in range(len(names))]

        for name, save_name in zip(names, save_names):
            data_tuple = []
            for i in range(2):
                if os.path.isfile(save_name[i]):
                    data_tuple.append(nib.load(save_name[i]).get_fdata())
            if len(data_tuple) == 2:
                dic[name] = data_tuple

    for db, name in zip(dbs, names):
        ps.append(Process(target=show_mean_mask, args=(db, name, dic, shape, shared_arrays[name], hist_build)))

    for p in ps:
        p.start()

    for p in ps:
        p.join()

    if hist_build:
        plt.figure(num="{} vs {} Histogram".format(names[0], names[1]))
        plt.subplot(1, 2, 1)
        plt.title(names[0] + " histogram")
        arr = np.array(shared_arrays[names[0]])
        arr = arr / arr.sum()
        print(names[0] + str(arr))
        plt.bar(range(len(arr)), arr)
        plt.subplot(1, 2, 2)
        plt.title(names[1] + " histogram")
        arr = np.array(shared_arrays[names[1]])
        arr = arr / arr.sum()
        print(names[1] + str(arr))
        plt.bar(range(len(arr)), arr)
        plt.show()
        plt.tight_layout()
        plt.savefig(os.path.join("data", "{}vs{}_hist.eps".format(names[0], names[1])), format='eps', dps=1000)
    else:
        plt.figure(num="Mean brain histogram")
        plt.subplot(1, 2, 1)
        plt.title(names[0] + " mean brain")
        plt.hist(dic[names[0]][0].flatten(), bins=100, density=True)
        plt.subplot(1, 2, 2)
        plt.title(names[1] + " mean brain")
        plt.hist(dic[names[1]][0].flatten(), bins=100, density=True)
        plt.figure(num="Mean hip histogram")
        plt.subplot(1, 2, 1)
        plt.title(names[0] + " mean hip")
        plt.hist(dic[names[0]][1].flatten(), bins=100, density=True)
        plt.subplot(1, 2, 2)
        plt.title(names[1] + " mean hip")
        plt.hist(dic[names[1]][1].flatten(), bins=100, density=True)

        plt.show()

        viewnii(dic[names[0]][0], mask=dic[names[0]][1], id=names[0], wait=0, label=names[0], border_only=False,
                quit_on_esc=False)
        viewnii(dic[names[1]][0], mask=dic[names[1]][1], id=names[1], wait=0, label=names[1], border_only=False,
                quit_on_esc=False)
        viewnii(np.zeros(shape), mask=dic[names[0]][1], ref=dic[names[1]][1], id='Center Crop Mask overlap', wait=0,
                border_only=False, label='{} and {} mean overlap'.format(names[0], names[1]))


def main():
    '''
    Runs if the module is called as a script (eg: python3 dataset.py <dataset_name> <frametime>)
    Executes self tests
    '''
    from transforms import ReturnPatch, Intensity, RandomFlip, Noisify, ToTensor, CenterCrop, RandomAffine
    train_transforms = Compose([ReturnPatch(patch_size=(32, 32)), RandomAffine(), Intensity(), RandomFlip(modes=['horflip']),
                                Noisify(), SoftTarget(), ToTensor()])  # default is 32 32 patch
    load_test = 'load_test' in argv
    itk_display = 'itk' in argv
    if load_test:
        print("Volumes will not be displayed due to load_test arg")
    dataset_name = argv[1] if len(argv) > 1 else "mnihip"
    wait = 10
    if 'wait' in argv:
        try:
            wait = int(argv[argv.index('wait') + 1])
        except IndexError:
            print("Please give a wait value when using wait arg")
    print("Dataset module running as script, executing dataset unit test in {}".format(dataset_name))

    if dataset_name == "fill_multitask":
        MultiTaskDataset.fill_hd_cache()
        dataset_name = "multitask"
    elif dataset_name == "hist_compare_nathip_harp":
        harp = NewHARP(fold=None, mode="all", verbose=True, transform=None)
        nathip = NatHIP(group='all', orientation=None, mode="all", verbose=True, transform=None, fold=None,
                        e2d=False, return_onehot=False)
        compare_annotations([harp, nathip], ["harp", "nathip"], None, hist_build=True)
    elif dataset_name == "compare_nathip_harp":
        shape = (160, 160, 160)
        harp = NewHARP(fold=None, mode="all", verbose=True, transform=CenterCrop(*shape))
        nathip = NatHIP(group='all', orientation=None, mode="all", verbose=True, transform=CenterCrop(*shape), fold=1,
                        e2d=False, return_onehot=False)
        compare_annotations([harp, nathip], ["harp", "nathip"], shape)
    elif dataset_name == "compare":
        shape = (160, 160, 160)
        mnihip = ConcatDataset([FloatHippocampusDataset(return_volume=True, transform=CenterCrop(*shape), orientation="coronal",
                                                        mode=m, verbose=False) for m in ["train", "validation", "test"]])
        harp = HARP("all", mode="all", mnireg=True, verbose=False, transform=CenterCrop(*shape))
        compare_annotations([mnihip, harp], ["mnihip", "harp"], shape)
    elif dataset_name == "process_newharp":
        NewHARP.process_rawvolumes()
        NewHARP.process_slices()
    elif dataset_name == "newharp_slices":
        # NewHARP.process_slices()
        # K fold test, assert no test volume in one fold is in another test
        if "test" in argv:
            for orientation in [None, "sagital", "coronal", "axial"]:
                for group in ["all", "ad", "mci", "cn"]:
                    print("Testing all folds of group: {}, orientation: {}".format(group, orientation))
                    folds = [NewHARP(mode="test", group=group, fold=i, verbose=False,
                             orientation=orientation) for i in range(1, 6)]
                    for i, fold in enumerate(folds):
                        ikeys = fold.keys
                        for j, compare_fold in enumerate(folds):
                            if j == i:
                                continue
                            to_compare_keys = compare_fold.keys
                            for ikey in ikeys:
                                assert not(ikey in to_compare_keys)
        else:
            unit_test(image_dataset=False, dataset="harp", hiponly=True, plt_show=True, nworkers=4, e2d=True)

    elif dataset_name == "compare_rawharp":
        shape = (160, 160, 160)
        mnihip = ConcatDataset([FloatHippocampusDataset(return_volume=True, transform=CenterCrop(*shape),
                                                        orientation="coronal", mode=m,
                                                        verbose=False) for m in ["train", "validation", "test"]])
        harp = HARP("all", mode="all", mnireg=False, verbose=False, transform=CenterCrop(*shape))
        compare_annotations([mnihip, harp], ["mnihip", "rawharp"], shape)
    elif dataset_name == "dti_displasia_volume":
        transform = Compose([ToTensor()])
        dti = DTISegDataset("test", use_t1=False, transform=transform, orientation=None, limit_masks=False, displasia=True,
                            balance='test', norm_type="zero_to_plus", patch_size=32, register_strategy='v02')
        for i in range(len(dti)):
            path, vol, mask = dti[i]
            print(path, vol.shape, mask.shape)
            viewnii(vol[0], mask[1], id=path, label="DTI", wait=0)
            viewnii(vol[1], mask[1], id=path, label="T1", wait=0)
            viewnii(vol[2], mask[1], id=path, label="T2", wait=0)

    elif dataset_name == "dti":
        transform = Compose([Intensity(), RandomAffine(), ToTensor()])
        for mode in ["train", "validation"]:
            DTISegDataset(mode, use_t1=False, transform=transform, orientation="sagital", limit_masks=False, displasia=True,
                          balance="510", norm_type="zero_to_plus", patch_size=32, register_strategy='v02').display_samples(10)
    elif dataset_name == "adni_slices":
        unit_test(image_dataset=False, dataset="adni", hiponly=True, plt_show=True, nworkers=4, e2d=True)
    elif dataset_name == "slices":
        unit_test(image_dataset=False, dataset="mnihip", hiponly=True, plt_show=True, nworkers=4, e2d=True)
    elif dataset_name == "harp_slices":
        unit_test(image_dataset=False, dataset="oldharp", hiponly=True, plt_show=True, nworkers=4, e2d=True)
    elif dataset_name == "concat":
        data_transforms = {'train': train_transforms, 'validation': train_transforms, 'test': Compose([CenterCrop(160, 160),
                                                                                                       ToTensor()])}
        mode = "train"
        data, dsizes = get_data(data_transforms=data_transforms, db="concat", e2d=True, batch_size=50 + 150*(mode != "test"))
        print("Dataset sizes: {}".format(dsizes))
        for o in orientations:
            batch = next(iter(data[o][mode]))
            display_batch(batch, o + " concat " + mode + " data")
        plt.show()
    elif dataset_name == "harp":
        group = None
        if 'group' in argv:
            try:
                group = argv[argv.index('group') + 1]
            except IndexError:
                print("Please give a group name when using group arg")
                group = None
        view_volumes(dataset_name, wait=wait, group=group, load_test=load_test, use_itk_snap=itk_display, split="split" in argv)
    else:
        view_volumes(dataset_name, wait=wait, load_test=load_test, use_itk_snap=itk_display, split="split" in argv)
    print("All tests completed!")


if __name__ == "__main__":
    main()
