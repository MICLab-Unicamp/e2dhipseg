'''
Some taks necessary to test quicknat predictions
'''

from os import path
from glob import glob
from sys import argv

import torch
import nibabel as nib
from tqdm import tqdm
import multiprocessing as mp

from nathip import nathip_path, NatHIP
from metrics import DICEMetric, EvalMetricCalculator


def loader(x):
    global dicer
    bname = path.basename(x).split('.')[0]

    pred = torch.from_numpy(nib.load(x).get_fdata()).float()
    try:
        tgt_path = glob(path.join(nathip_path, "final_dataset", "conformed_masks", "*{}*".format(bname)))[0]
    except IndexError:
        print(bname + " not in conformed masks")
        return
    tgt = torch.from_numpy(nib.load(tgt_path).get_fdata() > 127).float()

    return pred, tgt


if __name__ == "__main__":
    dicer = EvalMetricCalculator(DICEMetric())

    if '-p' in argv:
        group = "PACIENTES"
    elif '-c' in argv:
        group = "CONTROLES"
    else:
        group = "all"

    if "calculate" in argv:
        db = NatHIP(group=group, mode="all", fold=None, orientation=None, transform=None, e2d=False, return_onehot=False,
                    verbose=True)
        towork = db.get_volids()
        workers = mp.Pool(processes=4)  # more than this consumes more than 16 GB of memory

        quicknat_preds = []
        for work in towork:
            quicknat_preds += glob(path.join(nathip_path, "quicknat", "hip_only", "*{}.nii.gz".format(work)))

        assert len(quicknat_preds) > 0

        print("Calculating DICE for following predictions: {}".format(quicknat_preds))

        for pred, tgt in tqdm(workers.imap_unordered(loader, quicknat_preds), total=len(quicknat_preds)):
            dicer(pred.cuda(), tgt.cuda())

        results = dicer.final_results()

        print("Results: {}".format(results))

        with open(path.join(nathip_path, group + "test_quicknat_results.txt"), 'w') as test_file:
            test_file.write(str(results))

    elif "read" in argv:
        with open(path.join(nathip_path, group + "test_quicknat_results.txt"), 'r') as test_file:
            print(test_file.read())
    else:
        print("Give an arg.")
