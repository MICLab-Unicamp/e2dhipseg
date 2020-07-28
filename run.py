'''
Module with functions to run subprocesses and manage files

Also point of entry to run over external data
'''
from sys import argv
import sys
import traceback
import time
from datetime import date
import os
import json
from os.path import join as path_join
import shutil
import glob
import subprocess
from tqdm import tqdm
import nibabel as nib
import cv2 as cv
import numpy as np
from utils import viewnii, normalizeMri, file_dialog, alert_dialog, confirm_dialog, error_dialog, chunks
from dataset import mni_adni, default_adni, default_harp, mni_harp, HARP, HARP_CLASSES
from transforms import run_once, mni152reg, MNI_BUFFER_MATRIX_PATH, REGWorker
from get_models import get_models
from multiprocessing import Queue, Process, cpu_count

PRE_REGISTER_VOL_PATH = os.path.normpath('cache/pre_register_vol.nii.gz')
PRE_REGISTER_MASK_PATH = os.path.normpath('cache/pre_register_mask.nii.gz')

INVERSE_MATRIX_PATH = os.path.normpath('cache/invmnibuffer.mat')
TEMP_MASK_PATH = os.path.normpath('cache/mask.nii.gz')
MASKS_FOLDER = "e2dhipseg_masks"

if not os.path.isdir('cache'): os.mkdir('cache')
if not os.path.isdir(MASKS_FOLDER): os.mkdir(MASKS_FOLDER) 

def generate_stds():
    '''
    Walks through folders saving standard deviations (for old experiments without it)
    Folder hardcoded, this will run only once
    '''
    iterator = glob.iglob(os.path.join('/', "home", "diedre", "Dropbox", "anotebook", "models", "**", "*cons.txt"),
                          recursive=True)
    for f in iterator:
        dataset = os.path.basename(f).split('0.')
        if len(dataset) == 1:  # skipping old cons files
            continue
        print(dataset[0])
        print(f)
        with open(f, 'r') as fil:
            whole_file = fil.read()
            split_final_result = whole_file.split("Final result: ")
            mean = float(split_final_result[-1])
            consensus_list = np.array(json.loads(split_final_result[0])[-1])
            std = consensus_list.std()
            new_mean = consensus_list.mean()
            assert mean == new_mean

            print("consensus array: {}".format(consensus_list))
            print("saved mean: {} recalculated mean: {}".format(mean, new_mean))

            new_path = f.split(".txt")[0] + "_" + "std" + str(std) + ".txt"
            with open(new_path, 'a+') as new_fil:
                new_fil.write(whole_file)
                new_fil.write("std: {}".format(std))

        print("---------")


def hippodeep(folder="/home/diedre/git/hippodeep", display=False):
    '''
    Folder is where to look for .nii.gz files to run hippodeep
    '''
    with open("/home/diedre/git/diedre/logs/hippodeep_runs_{}.txt".format(time.ctime()), 'w') as logfile:
        for f in tqdm(glob.glob(os.path.join(folder, "*.nii.gz"))):
            try:
                path = os.path.basename(f)[:5]
                if path != "bruna" and path != "BRUNA":
                    print("Skipping {}".format(f))

                print(f)
                subprocess.run(["sh", "deepseg3.sh", os.path.basename(f)], stdout=logfile)
                if display:
                    result = nib.load(f[:-7] + "_mask_L.nii.gz").get_fdata() + nib.load(f[:-7] + "_mask_R.nii.gz").get_fdata()
                    result[result > 1] = 1
                    viewnii(normalizeMri(nib.load(f).get_fdata()), result)
                    cv.destroyAllWindows()
            except Exception as e:
                print("HIPPODEEP FATAL ERROR: {}".format(e))
                quit()


def dcm2niix(folder):
    '''
    Runs external dcm2niix utility on given folder
    '''
    subprocess.run(["/home/diedre/Downloads/NITRC-mricrogl-Downloads/mricrogl_linux/mricrogl_lx/dcm2niix", folder])


def freesurfer(folder, _format=".nii.gz", ncpu=None):
    '''
    Runs freesurfer run-all on a folder
    '''
    ncpus = cpu_count() if ncpu is None else ncpu
    to_process = glob.glob(path_join(folder, "*" + _format))
    number_of_jobs = len(to_process)
    assert number_of_jobs > 0
    print("Detected following volumes to process: {}".format(to_process))
    batch_size = number_of_jobs//ncpus
    if batch_size == 0:
        batch_size = 1
    print("Number of available threads: {}, batch size: {}.".format(ncpus, batch_size))
    batchs = chunks(to_process, batch_size)

    # Initialize workers
    workers = [Process(target=freesurfer_worker, args=(batch,)) for batch in batchs]

    print("Initialized {} workers for freesurfer processing.".format(len(workers)))

    # Start workers
    for worker in workers:
        worker.start()
    print("Started all workers.")

    # Wait for workers to finish
    for worker in tqdm(workers):
        worker.join()

    print("All workers done!")


def freesurfer_worker(batch):
    '''
    Freesurver run-all on all files on batch
    '''
    for vol in batch:
        vol_name = os.path.basename(vol)

        print("Transfering input to subjects folder...")
        pre_log = open(path_join("logs", vol_name + "_preprocess_freesurfer_worker_log{}.txt".format(str(date.today()))), 'wb')
        subprocess.run(["recon-all", "-i", vol, "-s", os.path.basename(vol)], stdout=pre_log)
        pre_log.close()

        print("Starting recon-all, this might take some hours.")
        recon_log = open(path_join("logs", vol_name + "_freesurfer_worker_log{}.txt".format(str(date.today()))), 'wb')
        subprocess.run(["recon-all", "-all", "-s", os.path.basename(vol)], stdout=recon_log)
        recon_log.close()


def nii2niigz(folder):
    '''
    Converts nii files in folder to nii.gz
    '''
    for f in tqdm(glob.iglob(os.path.join(folder, "*.nii"))):
        vol = nib.load(f)
        nib.save(vol, f + ".gz")


def invert_matrix(hip_path, ref_path, saved_matrix):
    '''
    Inverts the FSL matrix and returns hip to original state
    Returns path of final result
    '''
    
    if sys.platform == "win32":
        try: # when running frozen with pyInstaller
            flirt_executable = sys._MEIPASS+'\\flirt.exe' 
            convert_xfm_executable = sys._MEIPASS+'\\convert_xfm.exe' 
        except: # when running normally 
            flirt_executable = 'flirt.exe' 
            convert_xfm_executable = 'convert_xfm.exe'              
    else: 
        flirt_executable = 'flirt'   
        convert_xfm_executable = 'convert_xfm'        
    
    
    print("Inverting matrix... {}".format(hip_path))
    my_env = os.environ.copy(); my_env["FSLOUTPUTTYPE"] = "NIFTI_GZ" # set FSLOUTPUTTYPE=NIFTI_GZ
    subprocess.run([convert_xfm_executable, "-omat",  INVERSE_MATRIX_PATH, "-inverse", saved_matrix], env=my_env)
    print("Transforming back to original space...")
    subprocess.run([flirt_executable, "-in",  hip_path, "-ref", ref_path, "-out", "final_buffer.nii.gz", "-init", INVERSE_MATRIX_PATH,
                    "-applyxfm"], env=my_env)

    save_path = os.path.normpath(ref_path + "_voxelcount-{}_e2dhipmask.nii.gz".format(int(nib.load("final_buffer.nii.gz").get_fdata().sum())))

    try:
        shutil.move("final_buffer.nii.gz", save_path)
        os.remove(saved_matrix)
        os.remove(INVERSE_MATRIX_PATH)
        os.remove(hip_path)
    except OSError as oe:
        print("Error trying to remove post register matrix:")
        traceback.print_exception(oe)
        return

    print("Post-Registrations done.")
    return save_path


def adni_iso2mni(source_path=default_adni, save_path=mni_adni):
    '''
    Register all isometric ADNI volumes to MNI152
    '''
    print("Registering all adni volumes to mni152")
    source_masks_path = os.path.join(source_path, "masks")
    for s in tqdm(glob.glob(os.path.join(source_path, "samples", "*.nii.gz"))):
        mask_name = os.path.basename(s).split(".nii.gz")[0]
        mni152reg(s, os.path.join(source_masks_path, mask_name + ".nii.gz"),
                  save_path=os.path.join(save_path, "samples", mask_name),
                  mask_save_path=os.path.join(save_path, "masks", mask_name), remove=False, return_numpy=False)


def register_worker(q, save_path):
    while True:
        data = q.get()
        if data is None:
            return
        else:
            sample, mask, label, name = data
            reg_sample, reg_mask = reg_handler(sample, mask=mask, uid=str(os.getpid()))
            nib.save(nib.Nifti1Image(reg_sample, affine=None), path_join(save_path, "samples", HARP_CLASSES[label], str(name)))
            nib.save(nib.Nifti1Image(reg_mask, affine=None), path_join(save_path, "masks", HARP_CLASSES[label], str(name)))


def harp2mni(source_path=default_harp, save_path=mni_harp):
    '''
    Register all HARP volumes to MNI152
    '''
    print("Registering HARP volumes to mni152")
    harp = HARP("all", return_label=True, return_harp_id=True)
    ncpu = cpu_count() - 1  # leave one thread for main process to fill the queue / general use
    queue = Queue(maxsize=2)  # less things hanging in queue = less memory usage
    ps = []

    # Initialize register workers
    for i in range(ncpu):
        p = Process(target=register_worker, args=(queue, save_path))
        ps.append(p)

    # Start workers
    for p in ps:
        p.start()

    # Feed queue with all data to be registered
    for i in tqdm(range(len(harp))):
        sample, mask, label, harp_id = harp[i]
        queue.put((sample, mask, label, "{}.nii.gz".format(harp_id)))

    # Tell workers to stop
    for i in range(ncpu):
        queue.put(None)

    # Wait for workers to finish
    for p in ps:
        p.join()

    # Remove left over files
    for cache in glob.iglob(path_join("cache", "*.nii.gz")):
        try:
            print("Deleting {}".format(cache))
            os.remove(cache)
        except FileNotFoundError:
            print("File not found: {}".format(cache))
        except Exception as e:
            print("Error trying to cleanup cache {}".format(e))


def reg_handler(vol, mask=None, uid=''):
    '''
    Just calls the right registration processing function
    '''
    print("Registering input...")
    if mask is None:
        return reg_pre_post_single(vol, uid=uid)
    else:
        return reg_pre_post_pair(vol, mask, uid=uid)


def reg_pre_post_pair(vol, mask, uid=''):
    '''
    Pre and post processing of input volume and mask paths for mni152reg
    '''
    regworker = REGWorker(uid)
    begin = time.time()
    if type(vol) == np.ndarray and type(mask) == np.ndarray:
        volpath = regworker.add_worker_id(PRE_REGISTER_VOL_PATH)
        maskpath = regworker.add_worker_id(PRE_REGISTER_MASK_PATH)
        nib.save(nib.Nifti1Image(vol, affine=None), volpath)
        nib.save(nib.Nifti1Image(mask, affine=None), maskpath)
        vol = volpath
        mask = maskpath
    elif not(type(vol) == str) and not(type(mask) == str):
        raise ValueError("vol and mask should be a numpy volume or a path to the volume")

    print("Input PATHS -> Vol: {}\nMask: {}".format(vol, mask))
    vol, mask = mni152reg(vol, mask=mask, keep_matrix=True, worker_id=uid)

    if vol.max() > 1.0 or vol.min() < 0 or mask.max() > 1.0 or mask.min() < 0:
        print("WARNING: Data out of range, normalizing...")
        vol = normalizeMri(vol.astype(np.float32)).squeeze()
        mask = mask.astype(np.bool).astype(np.float32).squeeze()
    print("Registration took {}s".format(time.time() - begin))
    return vol, mask


def reg_pre_post_single(vol, uid=''):
    '''
    Pre and post processing of input volume for mni152reg
    '''
    regworker = REGWorker(uid)
    begin = time.time()
    # We want vol to be a path, but it can not be
    if type(vol) == np.ndarray:
        volpath = regworker.add_worker_id(PRE_REGISTER_VOL_PATH)
        nib.save(nib.Nifti1Image(vol, affine=None), volpath)
        vol = volpath
    elif not(type(vol) == str):
        raise ValueError("vol should be a numpy volume or a path to the volume")

    print("Input PATH -> Vol: {}".format(vol))
    vol = mni152reg(vol, mask=None, keep_matrix=True, worker_id=uid)

    if vol.max() > 1.0 or vol.min() < 0:
        print("Data out of range, normalizing...")
        vol = normalizeMri(vol.astype(np.float32)).squeeze()
    print("Registration took {}s".format(time.time() - begin))
    return vol


def main(runlist, models, reg, batch_mode, results_dst):
    '''
    Main loop to run externally
    '''
    for arg in tqdm(runlist):
        try:
            print("Processing {}\n".format(arg))
            if reg:
                # Register
                print("Performing MNI152 registration with FSL FLIRT due to -reg arg... This might take a few seconds...")
                volpath = arg
                arg = reg_handler(arg)

                # Segment
                vol, mask = run_once(None, models, numpy_input=arg, save=False, addch=True)

                # Register back
                nib.save(nib.Nifti1Image(mask, affine=None), TEMP_MASK_PATH)
                mask_path = invert_matrix(TEMP_MASK_PATH, volpath, MNI_BUFFER_MATRIX_PATH)
            else:
                print("Not doing pre-registration, using orientation detector instead.")

                vol, mask, mask_path = run_once(arg, models, save=True, return_mask_path=True, addch=True)

            try:
                check_file = os.path.join(results_dst, os.path.basename(mask_path))
                print("Saving result to {}".format(check_file))
                if os.path.isfile(check_file):
                    print("WARNING: Results for this volume already exist, overwriting...")
                    os.remove(check_file)
                shutil.move(mask_path, results_dst)
            except OSError as oe:
                msg = "Error moving results: {}, do you have the right permissions?".format(oe)
                print(msg)
                error_dialog(msg)
                quit()

            if "-display" in argv:
                print("Displaying results.")
                viewnii(vol, mask=mask)
        except Exception as e:
            traceback.print_exc()
            print("Error: {}, make sure your data is ok, and you have proper permissions. Please contact author in "
                  "https://github.com/dscarmo/e2dhipseg for issues".format(e))
            if batch_mode:
                print("Trying to continue... There might be errors in following segmentations.")


if __name__ == "__main__":
    mask = None
    try:
        arg = argv[1]
        folder = "/home/diedre/git/hippodeep"
        if len(argv) >= 4:
            mask = argv[3]
        if arg != "hippodeep" and len(argv) >= 3:
            folder = argv[2]
    except IndexError:
        arg = "run"

    if arg == "harp2mni":
        harp2mni()
    elif arg == "hippodeep":
        print("Due to author limitations, hippodeep must be run with terminal on the hippodeep folder, with the files on the same"
              "folder")
        hippodeep(folder)
    elif arg == "freesurfer":
        if "ncpu" in argv:
            ncpu = int(argv[-1])
        else:
            ncpu = None
        freesurfer(folder, ncpu=ncpu)
    elif arg == "dcm2niix":
        dcm2niix(folder)
    elif arg == "nii2niigz":
        nii2niigz(folder)
    elif arg == "mni152reg":
        print("Vol: {}\nMask: {}".format(folder, mask))
        vol, mask = reg_handler(folder, mask)
        viewnii(vol, mask, wait=0)
    elif arg == "adni2mni":
        adni_iso2mni()
    elif arg == "generatestd":
        generate_stds()
    else:
        batch_mode = False
        runlist = []
        if len(argv) == 1:
            alert_dialog("Please give a nift volume to run on.")
            arg = file_dialog()
            if arg is None:
                alert_dialog("No volume given, quitting.")
                quit()
            results_dst = os.path.join(os.path.dirname(arg), MASKS_FOLDER)
            os.makedirs(results_dst, exist_ok=True)
            print("Results will be in {}\n".format(os.path.join(arg, MASKS_FOLDER)))
            reg = confirm_dialog("Do you want to register the volume to MNI152 space? (recommended, can take a few seconds)")
        else:
            if "-dir" in argv:
                assert os.path.isdir(arg), "folder given in -d argument is not a folder!"
                results_dst = os.path.join(arg, MASKS_FOLDER)
                batch_mode = True
            else:
                results_dst = os.path.join(os.path.dirname(arg), MASKS_FOLDER)
                assert os.path.isfile(arg), ("File not found. Make sure the path for your nii input volume {} is correct. If its"
                                             "a directory use -dir".format(arg))

            os.makedirs(results_dst, exist_ok=True)
            print("Results will be in {}\n".format(os.path.join(arg, MASKS_FOLDER)))

            reg = "-reg" in argv
            print("Running pre-saved weights best model in {}".format(arg))
            
        if sys.platform == "win32":    
           try: weights_path = sys._MEIPASS+'\\weights' # when running frozen with pyInstaller
           except: weights_path = 'weights'            # when running normally
        else: weights_path = 'weights'
        
        models = get_models(False, True, True, False, True, False, dim='2d', model_folder=weights_path, verbose=False,
                            out_channels=2, apply_sigmoid=False, apply_softmax=True)

        if batch_mode:
            runlist = glob.glob(os.path.join(arg, "*.nii")) + glob.glob(os.path.join(arg, "*.nii.gz"))
            print("Running segmentation on the following files: {}\n".format(runlist))
            main(runlist, models, reg, batch_mode, results_dst)
        else:
            runlist.append(arg)
            main(runlist, models, reg, batch_mode, results_dst)
