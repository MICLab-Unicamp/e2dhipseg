'''
Main point of entry to perform all tasks on this code


Author: Diedre Carmo
https://github.com/dscarmo
'''
from sys import argv
from matplotlib import pyplot as plt
import numpy as np
import os
import glob
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.models as modelzoo
from unet import UNet, init_weights
from unet3d import Modified3DUNet
from dunet import get_dunet
import dataset
from dataset import orientations, default_adni, mni_adni
from metrics import DICELoss, DICEMetric
from train_evaluate import train_model, test_model, per_volume_test
from train_results import TrainResults
from transforms import CenterCrop, ToTensor, Compose, CenterCrop, Resize, ToNumpy, ReturnPatch, RandomFlip, Intensity, Noisify, RandomAffine
import multiprocessing as mp
from utils import check_name, parse_argv, plots

if __name__ == "__main__":
    plt.rcParams.update({'font.size': 16})

    display_volume, train, volume, db_name, notest, study_ths, wait, finetune = parse_argv(argv)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loss = DICELoss(apply_sigmoid=False)
    metric = DICEMetric(apply_sigmoid=False)

    #train_transforms = Compose([ReturnPatch(patch_size=(32, 32)), Intensity(), RandomFlip(modes=['horflip']), Noisify(), ToTensor()]) #default is 32 32 patch, arg* in sheet
    train_transforms = Compose([ReturnPatch(patch_size=(32, 32)), RandomAffine(), Intensity(), RandomFlip(modes=['horflip']), Noisify(), ToTensor()])
    data_transforms = {'train': train_transforms, 'validation': train_transforms, 'test': Compose([CenterCrop(160, 160), ToTensor()])}

    bnames = []
    ## PREVIOUS EXPERIMENTS LIST ##
    '''
    basename = "FLOAT0.01-DICE-120"
    bnames.append(basename)
    basename = "full-newdice-test"  # trained in full data
    bnames.append(basename)
    basename = "tPATCH320.001-VOLDICE-400" # patch approach, 32, flip only, only positive patch
    bnames.append(basename)
    basename = "INTFLIP-PATCH320.001-400"
    bnames.append(basename)
    basename = "RES-INTFLIP-PATCH320.001-400"
    bnames.append(basename)
    basename = "SMALL-RES-INTFLIP-PATCH160.001-1200" 
    bnames.append(basename)
    basename = "RES-INTFLIP-PATCH320.001-1200" # one of best, axial fail
    bnames.append(basename)
    basename = "RES-INTFLIP-newaug-PATCH320.001-1000" # looks very good, best axial performance
    bnames.append(basename)
    basename = "SMALL-RES-INTFLIP-newaug-PATCH160.001-1000"
    bnames.append(basename)

    basename = "RES-E2D-INTFLIP-nobias-PATCH32-ADAM-600" # last trained home, axial fail, divergence
    bnames.append(basename)
    basename = "RES-E2D-INTFLIP-nobias-PATCH32-0.001-600" # last trained lab, best, axial fail
    bnames.append(basename)

    bnames.append("INIT-RES-E2D-INTFLIP-nobias-PATCH32-ADAM-500")
    bnames.append("INIT-RES-NOBN-E2D-INTFLIP-nobias-PATCH32-0.005-500")
    bnames.append("INIT-RES-NOBN-E2D-INTFLIP-nobias-PATCH32-0.001-500")
    bnames.append("DEBUG-123-456") # TEST  
    bnames.append("DUNET-E2D-INTFLIP-PATCH32-0.005-400-finetuned0.00540030") #upper finetuned with ADAM '''

    # Bests
    #bnames.append("INIT-RES-E2D-INTFLIP-nobias-PATCH32-0.005-500") # best 
    #bnames.append("DUNET-E2D-INTFLIP-PATCH32-0.005-400") # dynamic unet

    # Fine Tune?
    #bnames.append("DUNET-E2D-INTFLIP-PATCH32-0.005-400-finetuned-0.000510030") # dynamic unet finetuned
    #bnames.append("INIT-RES-E2D-INTFLIP-nobias-PATCH32-0.005-500-finetuned") # vgg unet finetuned

    # Fast train
    #bnames.append("DUNET-E2D-INTFLIP-PATCH32-0.005-100") # dynamic unet 100 epoch less OF?
    #bnames.append("INIT-RES-E2D-INTFLIP-nobias-PATCH32-0.001-100") # 100 epoch less OF?

    # Mixed with ADNI
    #bnames.append("MIXED-INIT-RES-E2D-INTFLIP-nobias-PATCH32-0.001-200") # trying to mix with adni train 
    #bnames.append("MIXED-DUNET-E2D-INTFLIP-PATCH32-0.001-400") # dynamic unet trying to mix with adni train 
    #bnames.append("MIXED-INIT-newadni-RES-aff-E2D-INTFLIP-nobias-PATCH32-B600-0.005-500") # best mixed with adni

    #bnames.append("INIT-RES-E2D-INTFLIP-aff-nobias-PATCH32-B100-0.001-500") # best witch aff in mnihip
    #bnames.append("DUNET-E2D-INTFLIP-aff-PATCH32-B100-0.005-500") # dynamic unet with aff in mnihip

    # 3D Test
    #bnames.append("3D-ADAM-BASELINE-32PATCH")  # not extending to full brain 
    #bnames.append("3D-ADAM-BASELINE")  # decente 0.8 performance, best?
    #bnames.append("3D-BASELINE") # trying different LRs

    # 3D with aug
    #bnames.append("3D-ADAM-AUG")

    # My 3D UNet (both failed)
    #bnames.append("3D-ADAM-E3DUNET-BASELINE-SMALL-NOBN-nobias") # complete fail, there is no pt, dont test
    #bnames.append("3D-ADAM-32PATCH-E3DUNET-BASELINE-SMALL-NOBN-nobias") # trying 3dunet with patches

    # Modified 3D Unet with 2 channels input from E2D VGG11 Residual UNet
    #bnames.append("3D-ADAM-AUG-E3D") # awesome performance!
    #bnames.append("3D-ADAM-E3D") # similar to above
    #bnames.append("3D-ADAM-AUG-E3D-32PATCH") # Working slowly, test multi-cpu paralelism attempt, FAILED
    #bnames.append("3D-E3D-BASELINE") # SGD attempt

    # Retrain mixed adni with mniadni
    #bnames.append("MIXED-INIT-mniadni-RES-aff-E2D-INTFLIP-nobias-PATCH32-B600-0.005-500") # save for carapibus
    #bnames.append("MIXED-INIT-mniadni-RES-aff-E2D-INTFLIP-nobias-PATCH32-B100-0.005-500")
    ## END OF PREVIOUS EXPERIMENTS LIST ##

    # Were folders with past results are
    basepath = "/home/diedre/Dropbox/anotebook/models"


    def get_models(bias, e2d, res, small, bn, dunet, dim='2d', model_folder=None, verbose=True):
        '''
        Navigates through past results folder to load a past result
        '''
        if dim == '3d':
            model = UNet(1 + (e2d*2), 1, residual=res, small=small, bias=bias, bn=bn, dim=dim, verbose=verbose)
            if model_folder is not None:
                path = glob.glob(os.path.join(model_folder, "*.pt"))[0]
                model.load_state_dict(torch.load(path))
            return model
        else:
            models = {}
            for o in orientations:
                if dunet:
                    model = get_dunet()
                else:
                    model = UNet(1 + (e2d*2), 1, residual=res, small=small, bias=bias, bn=bn, dim=dim, verbose=verbose)
                if model_folder is not None:
                    path = glob.glob(os.path.join(model_folder, "*" + o + ".pt"))[0]
                    model.load_state_dict(torch.load(path))
                models[o] = model
            return models

    results = {}
    if len(bnames) == 0:
        print("Nothing to do, add a folder name to bnames")
        quit()
    for basename in bnames:
        print(basename)
        cpu_count = mp.cpu_count()
        model_folder = os.path.join(basepath, basename)
        
        print("Save point: {}".format(model_folder))

        bias = not check_name("nobias", basename)
        e2d = check_name("E2D", basename)
        res = check_name("RES", basename)
        small = check_name("SMALL", basename)
        adam = check_name("ADAM", basename)
        bn = not check_name("NOBN", basename)
        dunet = check_name("DUNET", basename)
        mixed = check_name("MIXED", basename)
        volumetric = check_name("3D", basename)
        aug3d = check_name("AUG", basename) and volumetric
        patch32 = check_name("32PATCH", basename)
        dim = '3d' if check_name("E3DUNET", basename) else '2d'
        e3d = check_name("E3D", basename)

        if volumetric:
            print('Chaing transforms to volumetric transforms')
            crop_shape = (160, 160, 160)
            
            if e3d:
                pre_saved_model = get_models(False, True, True, False, True, False, '2d', "weights")
                totensor = ToTensor(volumetric=volumetric, rune2d=True, e2d_device=device, pre_saved_model=pre_saved_model, small=patch32)
                cpu_count = 0
            else:
                pre_saved_model = None
                totensor = ToTensor(volumetric=volumetric)
                
            test_transforms = Compose([CenterCrop(160, 160, 160), totensor])

            if patch32:
                crop_shape = (32, 32, 32)
                if aug3d:
                    train_transforms = Compose([ReturnPatch(patch_size=crop_shape, kernel_shape=(3, 3)), Intensity(), Noisify(), totensor], time_trans=False)
                else:
                    train_transforms = Compose([ReturnPatch(patch_size=crop_shape, kernel_shape=(3, 3)), totensor])
            elif aug3d:
                print("Adding augs to 3D transforms")
                train_transforms = Compose([CenterCrop(*crop_shape), Intensity(), Noisify(), totensor])
            else:
                train_transforms = test_transforms

            data_transforms = {'train': train_transforms, 
                            'validation': train_transforms, 
                            'test': test_transforms}
        
        print("\nUsed transforms")
        for k, v in data_transforms.items():
            print("{} {}".format(k, v), end='\n\n')
        

        # Main volumetric performance test
        if volume:
            print("Calculating performance in volume")
            

            if db_name == "cc359":    
                test_dataset = dataset.CC359Data()
            elif db_name == "mnihip":
                test_dataset = dataset.FloatHippocampusDataset(mode="test", return_volume=True, verbose=False)
            elif db_name == "adni":
                split = (0.5, 0.1, 0.4) if mixed else (0, 0, 1)
                print("Adni split {}".format(split))
                test_dataset = dataset.FloatHippocampusDataset(h5path=default_adni, mode="test", adni=True, data_split=split, return_volume=True, mnireg=False)
            elif db_name == "mniadni":
                split = (0.5, 0.1, 0.4) if mixed else (0, 0, 1)
                print("MNI Adni split {}".format(split))
                test_dataset = dataset.FloatHippocampusDataset(h5path=mni_adni, mode="test", adni=True, data_split=split, return_volume=True, mnireg=True)
            else:
                raise ValueError("Invalid db_name {} for volume test".format(db_name))

            models = get_models(bias, e2d, res, small, bn, dunet, model_folder=model_folder)

            vol_result, ths_study, cons = per_volume_test(models, metric, data_transforms["test"], test_dataset, device, metric_name="DICE", display=display_volume, e2d=e2d, wait=wait, name=basename, study_ths=study_ths)
            results[basename] = vol_result
            plots(ths_study, cons, study_ths, opt=db_name, savepath=model_folder, mean_result=vol_result, name=basename)
            
        # Slice training and test
        else: 
            # Current default Hyperparameters
            NEPOCHS = 500
            LR = 0.005
            BATCH_SIZE = 2 if volumetric and not patch32 else 100
            hiponly = True
            print("Finetune: {}".format("yes" if finetune else "no"))
            shuffle = {x:True for x in ["train", "validation", "test"]}
            
            if mixed:
                print("Concatenating ADNI and Clarissa dataset")
                db_name = "concat"

            if volumetric:
                print("Using 3D network")
                db_name = "mnihip3d"
                hiponly = False
                e2d = False
            print("LR: {}, nepochs: {}, batch: {} db_name: {}".format(LR, NEPOCHS, BATCH_SIZE, db_name))
            
            print("Hiponly: " + str(hiponly))
            dataloaders, dataset_sizes = dataset.get_data(data_transforms, nworkers=cpu_count, batch_size=BATCH_SIZE, db=db_name, hiponly=hiponly, shuffle=shuffle, e2d=e2d, volumetric=volumetric)
            
            # Train from scracth or finetune 
            if train: 
                if volumetric:
                    if dim == '3d':
                        model = get_models(bias, e2d, res, small, bn, dunet, dim=dim)
                    else:
                        model = Modified3DUNet(1 + 1*e3d, 1)
                    model = model.to(device)
                    loss = DICELoss(apply_sigmoid=False, volumetric=True)
                    print("Replaced DICE Loss with volumetric one")

                    if adam or finetune:
                        print("Using adam")
                        opt = optim.Adam(model.parameters())
                    else:
                        opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
                    step_size = BATCH_SIZE//2
                    if step_size < 1: step_size = 1
                    scheduler = lr_scheduler.StepLR(opt, step_size=step_size, gamma=0.1)

                    train_model(model, loss, metric, opt, scheduler, dataloaders, device, dataset_sizes, 
                                inepoch_plot=False, loss_name="Batch DICE Loss", metric_name="Batch DICE", model_name=basename, num_epochs=NEPOCHS, 
                                save=True, savefolder=model_folder, patience=20)

                    print("Testing train result, filtering small components...")
                    test_model(model, loss, metric, dataloaders, dataset_sizes, device,
                            loss_name="Batch DICE Loss", metric_name="Batch DICE", filter_components=True, save_path=model_folder, db=db_name + "3dpost_train_test")
                else:
                    if finetune:
                        models = get_models(bias, e2d, res, small, bn, dunet, model_folder=model_folder)
                        model_folder += "-finetuned" + str(LR) + str(NEPOCHS) + str(BATCH_SIZE)
                        basename += "-finetuned" + str(LR) + str(NEPOCHS) + str(BATCH_SIZE)
                    else:
                        models = get_models(bias, e2d, res, small, bn, dunet)

                    for o in orientations:
                        print(o)
                        model = models[o]
                        
                        if check_name("INIT", basename):
                            print("initializing with vgg weights")
                            model.load_state_dict(init_weights(modelzoo.vgg11(pretrained=True), model))
                        
                        model = model.to(device)
                        
                        if adam or finetune:
                            print("Using adam")
                            opt = optim.Adam(model.parameters())
                        else:
                            opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
                        step_size = BATCH_SIZE//2
                        if step_size < 1: step_size = 1
                        scheduler = lr_scheduler.StepLR(opt, step_size=step_size, gamma=0.1)

                        train_model(model, loss, metric, opt, scheduler, dataloaders[o], device, dataset_sizes[o], 
                                    inepoch_plot=False, loss_name="Batch DICE Loss", metric_name="Batch DICE", model_name=basename + "-" + o, num_epochs=NEPOCHS, 
                                    save=True, savefolder=model_folder, patience=20)

                        print("Testing train result...")
                        test_model(model, loss, metric, dataloaders[o], dataset_sizes[o], device,
                                loss_name="Batch DICE Loss", metric_name="Batch DICE", save_path=model_folder, db=db_name + "2d_slice_test")
            # Only test and show results
            else: 
                print("Loading saved result and testing full volume with {}...".format(basename))
                plt.title("Loss in all orientations")
                if volumetric:
                    if dim == '3d':
                        model = get_models(bias, e2d, res, small, bn, dunet, dim=dim, model_folder=model_folder)    
                    else:
                        model = Modified3DUNet(1 +e3d*1, 1)
                    model = model.to(device)
                    model.load_state_dict(torch.load(glob.glob(os.path.join(model_folder, '*.pt'))[0]))
                    path = glob.glob(os.path.join(model_folder, '*.pkl'))[-1]
                    plot_path = os.path.join(model_folder, "plots.eps")
                    if os.path.isfile(plot_path):
                        print("Train plot already exists in: {}".format(plot_path))
                    else:
                        print("Display training results logged in {}".format(path))
                        loaded_results = TrainResults.load(path)
                        loaded_results.plot(show=False, loss_only=True, o='')
                        plt.tight_layout()
                        plt.savefig(plot_path, format='eps', dps=1000)
                        plt.show()

                    if not notest:
                        print("Testing train result, filtering small components...")
                        
                        '''test_model(model, loss, metric, dataloaders, dataset_sizes, device,
                                loss_name="Batch DICE Loss", metric_name="Batch DICE", filter_components=True, display_sample=display_volume, save_path=model_folder, db="mnihip3d")
                        print("----------------------------------------------------------------")
                        print("Testing again now in volumetric adni")
                        dataloaders["test"], dataset_sizes["test"] = dataset.get_adni_3d_dataloader(data_transforms["test"], cpu_count, mnireg=False)    
                        
                        test_model(model, loss, metric, dataloaders, dataset_sizes, device,
                                loss_name="Batch DICE Loss", metric_name="Batch DICE", filter_components=True, display_sample=display_volume, save_path=model_folder, db="isoadni3d")
                        print("----------------------------------------------------------------")'''
                        print("Testing again now in volumetric MNI adni")
                        dataloaders["test"], dataset_sizes["test"] = dataset.get_adni_3d_dataloader(data_transforms["test"], cpu_count, mnireg=True)    
                        
                        test_model(model, loss, metric, dataloaders, dataset_sizes, device,
                                loss_name="Batch DICE Loss", metric_name="Batch DICE", filter_components=True, display_sample=display_volume, save_path=model_folder, db="mniadni3d")
                else:
                    models = get_models(bias, e2d, res, small, bn, dunet, model_folder=model_folder)
                    for i, o in enumerate(orientations):
                        path = os.path.join(model_folder, basename + "-" + o + ".pkl")
                        loaded_results = TrainResults.load(path)
                        
                        loaded_results.plot(show=False, loss_only=True, o=o)

                        print(o)
                        if notest: continue
                    
                        models[o] = models[o].to(device)
                        test_model(models[o],
                                    loss, metric, dataloaders[o], dataset_sizes[o], device,
                                    loss_name="DICE Loss", metric_name="DICE")
                    plt.tight_layout()
                    plt.savefig(os.path.join(model_folder, "plots.eps"), format='eps', dps=1000)
                    plt.show()

    plt.show()
