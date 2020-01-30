'''
Get which slices have hippocampus presence, "standalone" script
'''
import glob
import os
import numpy as np
import time
import json
import multiprocessing as mp
from utils import get_slice, chunks
from dataset import multitask_hip_processed, add_path

workers = []
count_lock = mp.Lock()
orientations = ["sagital", "coronal", "axial"]


def fill_dict(file_range, nprocessed, full_dict):
    os.makedirs("/home/diedre/bigdata/Hippocampus/processed_slices", exist_ok=True)
    for fr in file_range:
        try:
            npz = np.load(fr)
        except Exception as e:
            print(e)
            quit()
        key = os.path.basename(fr).split('.')[0]

        vol, mask, orig = (npz['vol'].astype(np.float32), npz['mask'].astype(np.float32), npz['orig'].astype(np.float32))

        hip = mask[11] + mask[12]

        presence_dict = {'sagital': [], 'coronal': [], 'axial': []}

        for i, o in enumerate(orientations):
            for slice_index in range(hip.shape[i]):
                vol_slice = get_slice(vol, slice_index, o, rotate=90).astype(np.float16)
                hip_slice = get_slice(hip, slice_index, o, rotate=90).astype(np.uint8)
                mask_slice = get_slice(mask, slice_index, o, rotate=90).astype(np.uint8)
                orig_slice = get_slice(orig, slice_index, o, rotate=90).astype(np.uint8)
                if hip_slice.sum() != 0:
                    presence_dict[o].append(slice_index)
                    assert (vol_slice.dtype == np.dtype(np.float16) and mask_slice.dtype == np.dtype(np.uint8) and
                            orig_slice.dtype == np.dtype(np.uint8))
                    np.savez_compressed("/home/diedre/bigdata/Hippocampus/processed_slices/" + key + '_' + o[0] + str(slice_index)
                                        + ".npz", vol_slice=vol_slice, mask_slice=mask_slice, orig_slice=orig_slice)

        full_dict[key] = presence_dict

        nprocessed.value = nprocessed.value + 1


if __name__ == "__main__":
    nworkers = mp.cpu_count()//2
    nprocessed = mp.Value('i', 0)
    final_dict = mp.Manager().dict()

    print("nworkers {}".format(nworkers))

    files = glob.glob('/home/diedre/bigdata/Hippocampus/processed/*.npz')
    nfiles = len(files)
    for file_chunk in chunks(files, nfiles//nworkers):
        p = mp.Process(target=fill_dict, args=(file_chunk, nprocessed, final_dict))
        workers.append(p)
        p.start()

    while nprocessed.value != nfiles:
        time.sleep(1)
        print("{}/{}, {}%".format(nprocessed.value, nfiles, int(nprocessed.value*100/nfiles)), flush=True, end='\r')

    print("\nWaiting for processes to finish...")
    for worker in workers:
        worker.join()

    print("\nSaving presence dict...")
    with open(add_path(multitask_hip_processed, 'hip_presence.json'), 'w') as outfile:
        json.dump(dict(final_dict), outfile)

    print("\nDone.")
