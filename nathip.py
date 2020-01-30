'''
Handles our HCUnicamp dataset.
'''
import os
import multiprocessing as mp
import queue
import time
from sys import argv
import pickle

import torch
import nibabel as nib
import numpy as np
import glob
import h5py
from scipy.ndimage import zoom
import pandas as pd
from torch.utils import data
from utils import int_to_onehot, viewnii, one_hot
from shutil import copy2
from multiview import MultiViewer
from tqdm import tqdm

external_hd_path = os.path.join("/media", "diedre", "CSF Things", "bruna_manual", "Bruna_imagens_manuais", "Bruna_imagens")
nathip_path = os.path.join("/home", "diedre", "bigdata", "nathip")
nathip_drop_path = os.path.join("/home", "diedre", "Dropbox", "bigdata", "Hippocampus", "Bruna_imagens")
nathip_path = os.path.join("/home", "diedre", "Dropbox", "bigdata", "nathip")
destiny_path = os.path.join("/home", "diedre", "Dropbox", "bigdata", "Hippocampus", "volbrain")


class NatHIP(data.Dataset):
    '''
    Manual annotations of MNIHIP
    '''
    KEY_LIST = {'PACIENTES': ['200909251345', '200909301556', '201001060838', '201110100732', '201110211623', '201111180812',
                              '201201031305', '201302041513', '201305081437', '201305201200', '201307121700', '201307291741',
                              '201310091054', '201005261625', '201006160953', '201007160717', '201008270834', '201009291601',
                              '201210260720', '201211070851', '201212210948', '201301251102', '201302200709', '201303081647',
                              '201407251021', '201411211420', '201504271127', '201506190741', '201507250814', '201508141025',
                              '201011050840', '201102101331', '201102161644', '201103031408', '201103041056', '201304231333',
                              '201304260806', '201311190943', '201312060726', '201312161050', '201403210834', '201404020902',
                              '201404041001', '201404171503', '201004060908', '201004231438', '201105131519', '201108090803',
                              '201203061157', '201203231514', '201205021303', '201205171644', '201205180955', '201206121402',
                              '201207300912', '201208081146', '201407111104', '201407180933', '201407250725', '201407250816',
                              '201405301559', '201402281121', '201409230939', '201305201710', '201005121236', '201505231634',
                              '200911161134', '201601061306', '201410251404', '201005281730', '201409291101', '201504231331',
                              '201412181053', '201511051400', '201512020723', '201205160756', '201503041654', '201305240943',
                              '201607181310', '201005181125', '201504141828', '201006171334', '201505141137', '201210021625',
                              '201409171639', '201411270832', '201401171639', '201604121545', '201404171340', '201604291324',
                              '201003291022', '201407141217', '201507081811', '201101070709', '201601061521', '201107211451',
                              '201602050914', '201407261010', '201111301202', '201110041833', '201501301245', '201409190745',
                              '201511271334', '201410011758', '201504171317', '201507011003', '201405281449', '201509251203',
                              '201209100725', '201507080946', '201507170735', '201607201501', '201503110744', '201401241022',
                              '201408281419', '201110211657', '201401151601', '201103040953', '201507291610', '201410171025',
                              '201404081103', '201201300731', '201604251403', '201212050815', '201111210932', '201203141314',
                              '201311181040', '201003021154', '201412050729', '201508191604', '201601061447', '201602171247',
                              '201607141304', '201507221749', '201606141728'],
                'CONTROLES': ['200909011201', '200910211051', '200912020907', '201112211057', '201209051122', '201301301214',
                              '201307121053', '201005291021', '201008111720', '201009031037', '201210311022', '201211141058',
                              '201212120845', '201301090923', '201302061146', '201505161428', '201505201642', '201009301418',
                              '201011241222', '201012011302', '201312041335', '201402011405', '201402121517', '201406261439',
                              '201003291549', '201110071545', '201201031538', '201003101413', '201106031312', '201602170833',
                              '201102261238', '201011011335', '201103301351', '201204210836', '201308301233', '201305290834',
                              '201011241125', '201301301151', '201008061748', '201110190950', '201402251516', '201011011231',
                              '201011171149', '201102261153', '200910221801', '201207210819', '201008041451', '201005291312',
                              '201209191114', '201105191533', '201403291615', '201207201405', '201206231613', '201307231611',
                              '201207201346', '201302191714', '201611211411', '201508111758', '201111251016', '201003131148',
                              '201605181000', '201104011343', '201003131531', '201104151339']}

    # Software corrections to sheet labeling mistakes, last resort
    # INCLUDES NOISY SCANS
    # REMOVED_LIST = ['201409171639', '201005261625', '201112211057', '201011241125', '201411270832',
    #                 '201107211451', '201511051400', '201111301202', '201504231331', '201305201710',
    #                 '201501301245', '201404041001', '201105131519', '201404171340', '201005291021']

    # Only big mistakes
    REMOVED_LIST = ["201409171639", "201112211057", "201011241125", "201105131519", "201404171340", "201005291021",
                    "201404171340", "201602170833",  # no tag file
                    "200912020907", "200910211051"]  # wrong segs

    def __init__(self, **kwargs):
        super(NatHIP, self).__init__()

        # Sanitize mandatory args
        assert kwargs["group"] in ['all', 'CONTROLES', 'PACIENTES']
        assert kwargs["orientation"] in [None, "sagital", "coronal", "axial"]
        assert kwargs["mode"] in ["all", "train", "validation", "test"]
        assert kwargs["orientation"] is None, "slices not supported yet"

        # Extract kwargs, define properties
        self.group = kwargs["group"]
        self.orientation = kwargs["orientation"]
        self.mode = kwargs["mode"]
        self.e2d = kwargs["e2d"]
        fold = kwargs["fold"]
        self.return_onehot = kwargs["return_onehot"]
        self.folder = kwargs["folder"] if "folder" in kwargs else nathip_path
        self.return_fname = kwargs["return_fname"] if "return_fname" in kwargs else False
        self.verbose = kwargs["verbose"] if "verbose" in kwargs else False
        self.transform = kwargs["transform"] if "transform" in kwargs else None
        self.reconstruction_orientations = ["sagital", "coronal", "axial"]  # compatibility sake

        # Build file list
        if self.group == "all":
            self.subjects = NatHIP.KEY_LIST["PACIENTES"] + NatHIP.KEY_LIST["CONTROLES"]
        else:
            self.subjects = NatHIP.KEY_LIST[self.group]

        subject_len = len(self.subjects)
        train_len, test_len = int(subject_len*0.7), int(subject_len*0.2)

        # Fold Selection
        if fold in range(1, 6):
            self.train_val_set = self.subjects[:(fold-1)*test_len] + self.subjects[fold*test_len:]

            self.train_set = self.train_val_set[:train_len]
            self.val_set = self.train_val_set[train_len:]
            self.test_set = self.subjects[(fold-1)*test_len:fold*test_len]
        else:
            print("WARNING: Fold not selected! >.<")

        if self.mode == "train":
            self.subjects = self.train_set
        elif self.mode == "validation":
            self.subjects = self.val_set
        elif self.mode == "test":
            self.subjects = self.test_set
        elif self.mode == "all":
            # No changes to subjects
            pass

        # Software workaround wrong dataset
        for toremove in NatHIP.REMOVED_LIST:
            if toremove in self.subjects:
                self.subjects.remove(toremove)

        # Get exam paths from selected subjects
        self.keys = self.subjects

        # Add slices to keys if in slice mode
        if self.orientation is not None:
            raise NotImplementedError("Slice mode not implemented yet")

        print("NatHIP initialized with {}".format(kwargs))
        print(self.keys)
        print("{} volumes".format(len(self.keys)))

    def __len__(self):
        '''
        Returns length of the dataset
        '''
        return len(self.keys)

    def __getitem__(self, x):
        '''
        Decompress saved item and returns it
        '''
        subject = self.keys[x]

        if self.verbose:
            print(subject)

        if self.orientation is None:
            h5file = h5py.File(os.path.join(self.folder, "3d_h5dataset", "h5nathip.hdf5"), 'r')
            data_source = h5file[subject]
            img = data_source["img"][:]
        else:
            # TODO update to h5py slices
            raise NotImplementedError("Slices not implemented")
            if self.e2d:
                center_data = data_source["img"]
                slice_number = int(os.path.basename(subject).split('.')[0][1:])
                post_slice = os.path.join(os.path.dirname(subject), self.orientation[0] + str(slice_number + 1) + ".npz")
                pre_slice = os.path.join(os.path.dirname(subject), self.orientation[0] + str(slice_number - 1) + ".npz")

                if os.path.exists(post_slice):
                    post_data = np.load(post_slice)["img"]
                else:
                    post_data = center_data

                if os.path.exists(pre_slice):
                    pre_data = np.load(pre_slice)["img"]
                else:
                    pre_data = center_data

                img = np.zeros((3, center_data.shape[0], center_data.shape[1]), dtype=center_data.dtype)

                img[0] = pre_data
                img[1] = center_data
                img[2] = post_data
            else:
                img = data_source["img"][:]

        tgt = data_source["tgt"][:]

        h5file.close()

        if self.return_onehot:
            tgt = int_to_onehot(tgt)
            assert one_hot(torch.from_numpy(tgt).unsqueeze(0))

        if self.transform is not None:
            img, tgt = self.transform(img, tgt)

        if self.return_fname:
            return img, tgt, os.path.basename(subject)
        else:
            return img, tgt

    def get_volids(self):
        return self.subjects

    def get_by_name(self, name):
        for i in range(len(self)):
            if name in self.keys[i]:
                return self[i]


def get_group():
    if '-p' in argv:
        group = "PACIENTES"
    elif '-c' in argv:
        group = "CONTROLES"
    else:
        group = "all"

    print("Selected NatHIP group: {} group".format(group))

    return group


def display_dataset(dataset, exit_sign):
    '''
    GUI Process
    '''
    while True:
        data = dataset.get()
        if data is None:
            return
        else:
            volume, fname = data
            retcode = MultiViewer(volume, window_name=fname).display()
            if retcode == 'ESC':
                print("ESC captured.")
                exit_sign.value = 1
                return


def try_put(data, q, exit_sign, wait=0.1):
    success = False
    while not success:
        if exit_sign.value > 0:
            q.close()
            q.join_thread()
            return 'QUIT'

        try:
            q.put_nowait(data)
            success = True
        except queue.Full:
            time.sleep(wait)

    return 'OK'


def checkfiles(save_npz=False):
    dataset = mp.Queue(maxsize=20)
    exit_sign = mp.Value('i', 0)

    displayer = mp.Process(target=display_dataset, args=(dataset, exit_sign))
    displayer.start()

    for folder in glob.glob(os.path.join(nathip_drop_path, "IC2")):
        print("Walking thorugh folder {}".format(folder))
        for fpath in glob.glob(os.path.join(folder, "*.mnc")):
            print("\nReading {}...".format(fpath))
            tag_path = fpath.split('.')[0] + ".tag"

            # Read mnc file
            img = nib.load(fpath)
            img_data = img.get_fdata()
            img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
            img_affine = img.get_affine()

            tgt_data = np.zeros_like(img_data)

            # Read line by line points of tag file
            pts = []
            try:
                with open(tag_path) as tagfile:
                    line = tagfile.readline()
                    while line:
                        line_tokens = line.split(' ')

                        try:
                            point = np.array([int(line_tokens[1]), int(line_tokens[2]), int(line_tokens[3])], dtype=np.int)
                            # tgt_data[point[0], point[1], point[2]] = 1
                            pts.append(point)
                        except Exception:
                            pass

                        line = tagfile.readline()
            except FileNotFoundError as fnfe:
                print(fnfe)
                print("Skipping {}".format(fpath))
                continue

            npts = len(pts)
            print("Npoints: {}".format(npts))
            if npts < 1000:
                print("Skipping {} because segmentation is empty/weird".format(fpath))
                continue

            pts = nib.affines.apply_affine(img_affine, pts).astype(np.int)

            for pt in pts:
                tgt_data[pt[0], pt[1], pt[2]] = 1.0
            if save_npz:
                np.savez_compressed(fpath.split('.')[0] + ".npz", img=img_data, tgt=tgt_data, onehot=int_to_onehot(tgt_data))
            else:
                # Display results
                cube_side = 200
                volume_shape = img_data.shape
                zoom_factors = (cube_side/volume_shape[0], cube_side/volume_shape[1], cube_side/volume_shape[2])
                display = zoom(img_data, zoom_factors, order=2) + zoom(tgt_data, zoom_factors, order=0)
                display[display > 1] = 1
                data = (display, os.path.basename(fpath).split('.')[0])
                retcode = try_put(data, dataset, exit_sign)
                if retcode == 'QUIT':
                    print("Quitting early.")
                    displayer.join()
                    exit()

    dataset.put(None)
    dataset.close()
    dataset.join_thread()
    displayer.join()


def process_csvs(operation=None):
    '''
    Move manual segmentations to correct folder
    Intended structure
    Controls - Patients
    Foldernames with patient abbrv
    npz with full vol (date.npz), tag mask, one hot mask
    '''
    sjs = [pd.read_excel(os.path.join(nathip_path, "IC1_Editado.ods"), header=None, engine="odf").values,
           pd.read_excel(os.path.join(nathip_path, "IC2_Editado.ods"), header=None, engine="odf").values,
           pd.read_excel(os.path.join(nathip_path, "IC3_Editado.ods"), header=None, engine="odf").values]

    foundpair = 0
    os.makedirs("data/CONTROLES", exist_ok=True)
    os.makedirs("data/PACIENTES", exist_ok=True)

    # Sheet 1
    key_list = {'PACIENTES': [], 'CONTROLES': []}
    bruna_hash = []
    for i, sj in enumerate(sjs):
        for j, row in enumerate(sj):
            if i == 0:
                name = row[0]
                status = row[1]
            elif i == 1 or i == 2:
                name = row[1]
                if name[0] == "'":
                    name = name[1:-1]
                status = row[2]

            tokens = name.split('-')
            date = ''.join(tokens[:5])

            print(name, date, status)

            if status == 'P':
                dst_path = "data/PACIENTES/{}".format(date)
                key_list["PACIENTES"].append(date)
            elif status == 'C':
                dst_path = "data/CONTROLES/{}".format(date)
                key_list["CONTROLES"].append(date)
            else:
                continue

            if operation != "key_list":
                fmt = ".npz" if operation == "make_copy" else ".mnc"
                glob_string = os.path.join(nathip_drop_path, "IC{}".format(i + 1), "*_{}{}".format(j+1, fmt))
                print("Source: {}".format(glob_string), "Dest: {}".format(dst_path))
                if operation is None:
                    print("Not copied, just testing. Press anything to continue.")
                    input("")
                else:
                    try:
                        src = glob.glob(glob_string)[0]
                        foundpair += 1
                    except IndexError:
                        print("IC{} Bruna {} didnt have an npz".format(i + 1, j+1))
                        continue

                    if operation == "make_copy":
                        os.makedirs(dst_path, exist_ok=True)
                        dst = os.path.join(dst_path, date + ".npz")
                        copy2(src, dst)
                        print("Copied {} -> {}".format(src, dst))
                    elif operation == "make_hash":
                        original_name = os.path.splitext(os.path.basename(src))[0]
                        for items in bruna_hash:
                            if items[0] == original_name or items[1] == date:
                                raise ValueError("Duplicated entry in hash, something is wrong")

                        bruna_hash.append((original_name, date))
                        print("{}: {}".format(original_name, date))
                        print("Saveds {} hashes".format(len(bruna_hash)))

    print("Key list: {}".format(key_list))
    print("Found pair for {} brunas!".format(foundpair))
    if operation == "make_hash":
        save_hash = os.path.join("data", "bruna_hash.pkl")
        print("Saving hash in {}".format(save_hash))
        with open(save_hash, "wb") as save_pkl:
            pickle.dump(bruna_hash, save_pkl)


if __name__ == "__main__":

    if "check" in argv:
        checkfiles(save_npz=True)
    elif "process" in argv:
        operation = None
        if "-op" in argv:
            try:
                operation = argv[argv.index("-op") + 1]
            except IndexError:
                print("When using -op give something to do.")
        process_csvs(operation=operation)
    elif "display" in argv:
        print("Testing all modes and folds...")
        for mode in ["train", "validation", "test"]:
            for fold in range(1, 6):
                print(mode, fold)
                db = NatHIP(group="all", return_fname=True, mode=mode, orientation=None, fold=fold, verbose=True, e2d=False,
                            return_onehot=False)
                print(db.subjects)
                print('*'*10)

        for group in ["CONTROLES", "PACIENTES"]:
            db = NatHIP(group=group, return_fname=True, mode="all", orientation=None, fold=None, verbose=True, e2d=False,
                        return_onehot=False)

            print(len(db))

            itc = iter(db)
            for img, tgt, fname in tqdm(itc):
                if "load_test" in argv:
                    continue
                viewnii(img, mask=tgt, wait=1, id=group, label=fname,
                        border_only=False, rotate=0)
    else:
        print("No args given.")
