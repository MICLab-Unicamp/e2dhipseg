'''
Main point of entry to perform all experimental tasks

Handlings saving files and results of experiments
This is quite complicated code due to being backwards compatible with very old experiments


Author: Diedre Carmo
https://github.com/dscarmo
'''
from sys import argv
from matplotlib import pyplot as plt
import os
import glob
import time
import numpy as np
import torch
import torchvision.models as modelzoo
from unet import init_weights
from unet3d import Modified3DUNet
import dataset
from dataset import orientations, default_adni, mni_adni, default_harp, get_harp_group
from metrics import DICELoss, DICEMetric, CELoss, JointLoss, JointMetric, GeneralizedDice, BoundaryLoss
from train_evaluate import train_model, test_model, per_volume_test, get_optimizer_scheduler
from train_results import TrainResults
from transforms import ToTensor, Compose, CenterCrop, get_data_transforms
import multiprocessing as mp
from utils import (parse_argv, plots, get_device, get_memory_state, half_multi_task_labels, multi_task_labels,
                   HALF_MULTI_TASK_NCHANNELS, MULTI_TASK_NCHANNELS)
from get_models import get_models
from experiments import experiments, check_experiment
from nathip import NatHIP, get_group


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 16})

    cmd_args = parse_argv(argv)

    device = get_device()

    print("Goin to work on the following folders: {}".format(experiments))

    # Were folders with past results are
    basepath = "/home/diedre/Dropbox/anotebook/models"

    memory_state = get_memory_state()
    results = {}

    if len(experiments) == 0:
        print("Nothing to do, add a folder name to experiments")
        quit()

    fold_means = []

    for experiment in experiments:
        print(experiment)
        cpu_count = mp.cpu_count()//memory_state["memory_factor"]
        print("Intended worker count: {}".format(cpu_count))
        model_folder = os.path.join(basepath, experiment)

        print("Save point: {}".format(model_folder))

        args = check_experiment(experiment)

        if (args["gdl"] or args["boundary"]) and args["multitask"]:
            out_channels = (args["multitask"]*(8 if args["NLR"] else 16)) + 1
        else:
            out_channels = ((not (args["gdl"] or args["boundary"]))*(args["multitask"]*(8 if args["NLR"] else 16)) + 1 +
                            int(args["gdl"] or args["boundary"]))

        if args["3CH"]:
            out_channels = 2
            print("Out channels for e2d model: {}".format(out_channels))
        else:
            print("Out channels: {}".format(out_channels))

        if args["multitask"]:
            if args["NLR"]:
                labels = half_multi_task_labels
                nlabels = HALF_MULTI_TASK_NCHANNELS
            else:
                labels = multi_task_labels
                nlabels = MULTI_TASK_NCHANNELS
        else:
            labels = ["other", "hippocampus"]
            nlabels = 2

        if args["e2d"]:
            print("Using E2D")

        apply_softmax = False

        if args["ce"]:
            print("Replacing loss with crossentropyloss and deactivating sigmoid")
            loss = CELoss(nlabels, device)
            metric = DICEMetric(apply_sigmoid=False, skip_ths=(args["softarget"] and (not args["volumetric"])))
            metric_name = "DICE"
            sigmoid = False
            loss_name = "Cross Entropy Loss"
        elif args["gdl"]:
            print("Replacing loss with GDL and deactivating sigmoid, APPLYING SOFTMAX")
            loss = GeneralizedDice(idc=[1, ] + [1 for _ in range(nlabels - 1)], loss=True)
            metric = DICEMetric(skip_ths=True, one_hot_one_class=(nlabels == 2))
            metric_name = "Dice"
            sigmoid = False
            apply_softmax = True
            loss_name = "GDL"
        elif args["boundary"]:
            print("Replacing loss with Boundary Loss and deactivating sigmoid, APPLYING SOFTMAX")
            loss = BoundaryLoss(idc=[1, ] + [1 for _ in range(nlabels - 1)], max_ncalls=args["nepochs"], init_weights=[1, 0],
                                use_gdl=not args["DLBound"])
            metric = DICEMetric(skip_ths=True, one_hot_one_class=(nlabels == 2))
            metric_name = "Dice"
            sigmoid = False
            apply_softmax = True
            loss_name = "Boundary"
        elif args["classify"]:  # this did not work at all
            print("Joint Loss and Joint Metric for classification!")
            loss = JointLoss(apply_sigmoid=False, dice_weight=10)
            metric = JointMetric(apply_sigmoid=False, skip_ths=(args["softarget"] and (not args["volumetric"])))
            metric_name = "DICE/ACC"
            loss_name = "Joint Loss"
            sigmoid = True
        else:
            loss = DICELoss(apply_sigmoid=False, multi_target=args["multitask"], negative_loss=args["multitask"])
            metric = DICEMetric(apply_sigmoid=False, skip_ths=(args["softarget"] and (not args["volumetric"])))
            metric_name = "DICE"
            sigmoid = True
            loss_name = "Batch DICELoss"

        if args["e3d"]:
            cpu_count = 0
            pre_saved_model = get_models(False, True, True, False, True, False, '2d',
                                         "weights_harp" if args["hweights"] else "weights", out_channels=out_channels,
                                         apply_sigmoid=sigmoid, classify=args["classify"])
        else:
            pre_saved_model = None
        data_transforms = get_data_transforms(args, pre_saved_model=pre_saved_model, device=device, nlabels=nlabels)

        # Volumetric performance test
        if cmd_args["volume"]:
            print("Calculating performance in volume")

            if cmd_args["db_name"] == "nathip":
                group = get_group()
                test_dataset = NatHIP(group=group, orientation=None, mode="all", verbose=True, transform=None, fold=1,
                                      e2d=False, return_onehot=False)
            elif cmd_args["db_name"] == "cc359":
                test_dataset = dataset.CC359Data()
            elif cmd_args["db_name"] == "mnihip":
                test_dataset = dataset.FloatHippocampusDataset(mode="test", return_volume=True, verbose=False)
            elif cmd_args["db_name"] == "adni":
                split = (0.5, 0.1, 0.4) if args["mixed"] else (0, 0, 1)
                print("Adni split {}".format(split))
                test_dataset = dataset.FloatHippocampusDataset(h5path=default_adni, mode="test", adni=True, data_split=split,
                                                               return_volume=True, mnireg=False)
            elif cmd_args["db_name"] == "mniadni":
                split = (0.5, 0.1, 0.4) if args["mixed"] else (0, 0, 1)
                print("MNI Adni split {}".format(split))
                test_dataset = dataset.FloatHippocampusDataset(h5path=mni_adni, mode="test", adni=True, data_split=split,
                                                               return_volume=True, mnireg=True)
            elif cmd_args["db_name"] == "oldharp":
                group = 'all'
                print("Harp using group: {}".format(group))
                test_dataset = dataset.FloatHippocampusDataset(h5path=default_harp, harp=True, adni=False,
                                                               data_split=(0.7, 0.1, 0.2), mode="test", return_volume=True,
                                                               verbose=True, mnireg=False, return_label=args["classify"])
            elif cmd_args["db_name"] == "harp":
                test_dataset = dataset.NewHARP(fold=args["FOLD"], mode="test", group=get_harp_group())
            elif cmd_args["db_name"] == "multitask":
                test_dataset = dataset.MultiTaskDataset(mode='test', verbose=False, hiponly=True, dim='3d',
                                                        return_onehot=not args["ce"], merge_left_right=False,
                                                        zero_background=args["ZB"])
            else:
                raise ValueError("Invalid db_name {} for volume test".format(cmd_args["db_name"]))

            models = get_models(args["bias"], args["e2d"], args["res"], args["small"], args["bn"], args["dunet"],
                                model_folder=model_folder, out_channels=out_channels, apply_sigmoid=sigmoid,
                                classify=args["classify"], apply_softmax=apply_softmax)

            vol_result, ths_study, cons, std, eval_ = per_volume_test(models, DICEMetric(apply_sigmoid=False),
                                                                      Compose([CenterCrop(160, 160), ToTensor()]), test_dataset,
                                                                      device, metric_name=metric_name,
                                                                      display=cmd_args["display_volume"],
                                                                      e2d=args["e2d"], wait=cmd_args["wait"], name=experiment,
                                                                      study_ths=cmd_args["study_ths"], ce_output=args["ce"],
                                                                      classify=args["classify"], rot=cmd_args["rot"])
            fold_means.append(vol_result)
            results[experiment] = vol_result
            plots(ths_study, cons, cmd_args["study_ths"], opt=cmd_args["db_name"], savepath=model_folder, mean_result=vol_result,
                  name=get_harp_group() + get_group() + '_' + experiment, std_result=std, results=eval_)

        # Slice training and test
        else:
            # Current default Hyperparameters
            nepochs, batch_size = args["nepochs"], args["batch_size"]
            if nepochs is None:
                NEPOCHS = 500
            else:
                NEPOCHS = nepochs
            if args["NOP"]:
                patience = 0
            else:
                patience = NEPOCHS//5
            print("Patience {}".format(patience))

            if cmd_args["lr"] is None:
                LR = args["LR"]
                if args["adam"]:
                    if args["LR"] is None:
                        print("LR not defined for adam, using 0.0001")
                        LR = 0.0001
            else:
                LR = cmd_args["lr"]

            if args["batch_size"] is None:
                if cmd_args["batch_size"] is None:
                    big_input = (args["center"] or args["center_halfaug"] or args["center_fullaug"])
                    if big_input:
                        print("Big input detected, smaller batch size")
                        BATCH_SIZE = 40
                    else:
                        BATCH_SIZE = 2 if args["volumetric"] and not args["patch32"] else 200
                else:
                    BATCH_SIZE = cmd_args["batch_size"]
            else:
                BATCH_SIZE = args["batch_size"]

            hiponly = True
            print("Finetune: {}".format("yes" if cmd_args["finetune"] else "no"))
            shuffle = {x: True for x in ["train", "validation", "test"]}

            if args["mixed"]:
                print("Concatenating ADNI and MNIHIP dataset")
                cmd_args["db_name"] = "concat"
            elif args["mixharp"]:
                print("Concatenating HARP and MNIHIP dataset!")
                cmd_args["db_name"] = "mixharp"
            elif args["harp"]:
                print("Using HARP dataset for training")
                cmd_args["db_name"] = "harp"
            elif args["oldharp"]:
                print("Using old harp dataset for training")
                cmd_args["db_name"] = "oldharp"
            elif args["multitask"]:
                print("Using multitask dataset")
                cmd_args["db_name"] = "multitask"
            else:
                cmd_args["db_name"] = "mnihip"

            if args["volumetric"]:
                print("Using 3D network")
                if cmd_args["db_name"] == "mnihip":
                    cmd_args["db_name"] = "mnihip3d"
                elif cmd_args["db_name"] == "harp":
                    if args["FOLD"] is not None:
                        cmd_args["db_name"] = "harp3dfold"
                    else:
                        cmd_args["db_name"] = "harp3d"
                hiponly = False
                e2d = False

            print("Hiponly: " + str(hiponly))

            print(("Please check augmentation parameters: "
                  "Train transform: {}\nValidation transforms: {}\nTest transforms: {}\n").format(data_transforms["train"],
                                                                                                  data_transforms["validation"],
                                                                                                  data_transforms["test"]))
            print("Please check the arguments bellow:\nCommand line arguments: {}\n\nExperiment arguments: {}\n".format(
                cmd_args, args
            ))
            print("LR: {}, nepochs: {}, batch: {} db_name: {}\n".format(LR, NEPOCHS, BATCH_SIZE, cmd_args["db_name"]))
            if not cmd_args["nowait"]:
                for s in range(5, 0, -1):
                    print("{}...".format(s))
                    time.sleep(1)

            if cmd_args["train"] or (not cmd_args["notest"]):
                print("Preparing datasets and getting first batch...")
                dataloaders, dataset_sizes = dataset.get_data(data_transforms, nworkers=cpu_count, batch_size=BATCH_SIZE,
                                                              db=cmd_args["db_name"], hiponly=hiponly, shuffle=shuffle,
                                                              e2d=args["e2d"], volumetric=args["volumetric"], nlr=args["NLR"],
                                                              ce=args["ce"], classify=args["classify"],
                                                              gdl=args["gdl"] or args["boundary"], fold=args["FOLD"])
                print("Check output of dataset initialization! ^ Is it what is expected?\n")
                print("Reading training engines...")
                if not cmd_args["nowait"]:
                    for s in range(5, 0, -1):
                        print("{}...".format(s))
                        time.sleep(1)
                print("\n--- Fire! ---\n")

            # Train from scracth or finetune
            if cmd_args["train"]:
                if args["volumetric"]:
                    if args["dim"] == '3d':
                        model = get_models(args["bias"], args["e2d"], args["res"], args["small"], args["bn"], args["dunet"],
                                           dim=args["dim"], out_channels=out_channels, apply_sigmoid=sigmoid,
                                           classify=args["classify"])
                    else:
                        model = Modified3DUNet(1 + int(args["e3d"]), 1)
                    model = model.to(device)
                    loss = DICELoss(apply_sigmoid=False, volumetric=True)
                    print("Replaced Loss with volumetric one")

                    opt, scheduler = get_optimizer_scheduler(model, args, LR=LR, patience=patience, NEPOCHS=NEPOCHS)

                    train_model(model, loss, metric, opt, scheduler, dataloaders, device, dataset_sizes,
                                inepoch_plot=False, loss_name=loss_name, metric_name=metric_name, model_name=experiment,
                                num_epochs=NEPOCHS, save=True, savefolder=model_folder, patience=patience,
                                classify=args["classify"])

                    print("Testing train result, filtering small components...")
                    test_model(model, loss, metric, dataloaders, dataset_sizes, device,
                               loss_name=loss_name, metric_name=metric_name, filter_components=True, save_path=model_folder,
                               db=cmd_args["db_name"] + "3dpost_train_test", classify=args["classify"])
                else:
                    if cmd_args["finetune"]:
                        models = get_models(args["bias"], args["e2d"], args["res"], args["small"], args["bn"], args["dunet"],
                                            model_folder=model_folder, out_channels=out_channels, apply_sigmoid=sigmoid,
                                            classify=args["classify"])
                        model_folder += "-finetuned" + str(LR) + str(NEPOCHS) + str(BATCH_SIZE)
                        experiment += "-finetuned" + str(LR) + str(NEPOCHS) + str(BATCH_SIZE)
                    else:
                        models = get_models(args["bias"], args["e2d"], args["res"], args["small"], args["bn"], args["dunet"],
                                            out_channels=out_channels, apply_sigmoid=sigmoid, classify=args["classify"],
                                            apply_softmax=apply_softmax)

                    for o in orientations:
                        print(o)
                        model = models[o]

                        if args["INIT"]:
                            print("initializing with vgg weights")
                            model.load_state_dict(init_weights(modelzoo.vgg11(pretrained=True), model))

                        model = model.to(device)

                        opt, scheduler = get_optimizer_scheduler(model, args, LR=LR, patience=patience, NEPOCHS=NEPOCHS,
                                                                 finetune=cmd_args["finetune"])

                        if args["gdl"] or args["boundary"]:
                            print("GDL/Boundary Loss ON")

                        if args["boundary"]:
                            loss.reset_weights()

                        train_model(model, loss, metric, opt, scheduler, dataloaders[o], device, dataset_sizes[o],
                                    inepoch_plot=False, loss_name=loss_name, metric_name=metric_name,
                                    model_name=experiment + "-" + o, num_epochs=NEPOCHS, save=True, savefolder=model_folder,
                                    patience=patience, classify=args["classify"], probs2onehot=args["gdl"] or args["boundary"])

                        print("Testing train result...")
                        test_model(model, loss, metric, dataloaders[o], dataset_sizes[o], device, loss_name=loss_name,
                                   metric_name=metric_name, save_path=model_folder, db=cmd_args["db_name"] + o + "2d_slice_test",
                                   classify=args["classify"], probs2onehot=args["gdl"] or args["boundary"])
            # Only test and show results
            else:
                print("Loading saved result and testing full volume with {}...".format(experiment))
                if args["volumetric"]:
                    if args["dim"] == '3d':
                        model = get_models(args["bias"], args["e2d"], args["res"], args["small"], args["bn"], args["dunet"],
                                           dim=args["dim"], model_folder=model_folder, out_channels=out_channels,
                                           apply_sigmoid=sigmoid, classify=args["classify"])
                    else:
                        model = Modified3DUNet(1 + args["e3d"]*1, 1)
                    model = model.to(device)
                    model.load_state_dict(torch.load(glob.glob(os.path.join(model_folder, '*.pt'))[0]))
                    try:
                        path = glob.glob(os.path.join(model_folder, '*.pkl'))[-1]
                    except IndexError:
                        print("ERROR: PKL train results file not found. Skipping experiment")
                        continue
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

                    if not cmd_args["notest"]:
                        '''print("Testing train result in mnihip, filtering small components...")
                        dataloaders, dataset_sizes = dataset.get_data(data_transforms, nworkers=cpu_count, batch_size=BATCH_SIZE,
                                                                      db="mnihip3d", hiponly=hiponly, shuffle=shuffle,
                                                                      e2d=args["e2d"], volumetric=args["volumetric"],
                                                                      classify=args["classify"])
                        test_model(model, loss, metric, dataloaders, dataset_sizes, device, loss_name=loss_name,
                                   metric_name=metric_name, filter_components=True, display_sample=cmd_args["display_volume"],
                                   save_path=model_folder, db="mnihip3d", classify=args["classify"])
                        print("----------------------------------------------------------------")

                        print("Testing again now in volumetric adni")
                        dataloaders["test"], dataset_sizes["test"] = dataset.get_adni_3d_dataloader(data_transforms["test"],
                                                                                                    cpu_count, mnireg=False)

                        test_model(model, loss, metric, dataloaders, dataset_sizes, device, loss_name=loss_name,
                                   metric_name=metric_name, filter_components=True, display_sample=cmd_args["display_volume"],
                                   save_path=model_folder, db="isoadni3d", classify=args["classify"])
                        print("----------------------------------------------------------------")

                        print("Testing again now in volumetric MNI adni")
                        dataloaders["test"], dataset_sizes["test"] = dataset.get_adni_3d_dataloader(data_transforms["test"],
                                                                                                    cpu_count, mnireg=True)

                        test_model(model, loss, metric, dataloaders, dataset_sizes, device, loss_name=loss_name,
                                   metric_name=metric_name, filter_components=True, display_sample=cmd_args["display_volume"],
                                   save_path=model_folder, db="mniadni3d", classify=args["classify"])
                        print("----------------------------------------------------------------")

                        print("Testing again now in Harp3D")
                        dataloaders, dataset_sizes = dataset.get_data(data_transforms, db="harp3d", volumetric=True,
                                                                      hiponly=False, batch_size=BATCH_SIZE, nworkers=cpu_count,
                                                                      classify=args["classify"])

                        print("Testing in HARP3DFOLD")
                        dataloaders, dataset_sizes = dataset.get_data(data_transforms, db="harp3dfold", volumetric=True,
                                                                      hiponly=False, batch_size=BATCH_SIZE, nworkers=cpu_count,
                                                                      classify=args["classify"], fold=args["FOLD"])

                        test_model(model, loss, metric, dataloaders, dataset_sizes, device, loss_name=loss_name,
                                   metric_name=metric_name, filter_components=True, display_sample=cmd_args["display_volume"],
                                   save_path=model_folder, db="harp3d", classify=args["classify"])'''

                        dataloaders, dataset_sizes = dataset.get_data(data_transforms, db="nathip", volumetric=True,
                                                                      hiponly=False, batch_size=BATCH_SIZE, nworkers=cpu_count,
                                                                      classify=args["classify"], fold=None)

                        results = test_model(model, loss, metric, dataloaders, dataset_sizes, device, loss_name=loss_name,
                                             metric_name=metric_name, filter_components=True,
                                             display_sample=cmd_args["display_volume"], save_path=model_folder, db="nathip",
                                             classify=args["classify"])

                        with open(os.path.join(model_folder, get_group() + "eval_metrics.txt"), 'a') as f:
                            f.write(results)

                else:
                    if not cmd_args["notest"]:
                        models = get_models(args["bias"], args["e2d"], args["res"], args["small"], args["bn"], args["dunet"],
                                            model_folder=model_folder, out_channels=out_channels, apply_sigmoid=sigmoid,
                                            classify=args["classify"])

                    for i, o in enumerate(orientations):
                        path = os.path.join(model_folder, experiment + "-" + o + ".pkl")
                        loaded_results = TrainResults.load(path)

                        if getattr(loaded_results, "metric_name_class", False) is False and args["classify"]:
                            print("Wrongly formated loaded results, adding metric_name_class")
                            setattr(loaded_results, "metric_name_class", "Class Accuracy")

                        loaded_results.plot(show=False, loss_only=False, o=o, generate_new_figure=cmd_args["multifigures"],
                                            ylim=1.0, classify=args["classify"], lower_ylim=0.5)

                        if cmd_args["notest"]:
                            continue

                        models[o] = models[o].to(device)
                        test_model(models[o], loss, metric, dataloaders[o], dataset_sizes[o], device, loss_name=loss_name,
                                   metric_name=metric_name, filter_components=False, display_sample=cmd_args["display_volume"],
                                   save_path=model_folder, db="multitask_slice", classify=args["classify"])

                    plt.tight_layout()
                    plt.savefig(os.path.join(model_folder, "plots.eps"), format='eps', dps=1000)
                    print("Displaying: " + os.path.basename(model_folder))
                    plt.show()
        print("Finished main scripts on {}".format(experiment))

    # Save mean and STD of 5FOLD experiments
    if len(fold_means) > 0 and len(fold_means) == 5:
        std = np.array(fold_means).std()
        mean = np.array(fold_means).mean()
        print("Fold Means: {}".format(fold_means))
        print("Mean of all volumetric experiments: {}\nSTD: {}".format(mean, std))
        with open(os.path.join(basepath, str(experiments[-1]) + "_MEAN_" + str(mean) + "_STD_" + str(std) + ".txt"), 'a') as f:
            f.write(str(mean))
            f.write(str(std))
            f.write(str(fold_means))
    elif len(fold_means) != 5:
        print("Not saving fold STD and MEAN.")
    plt.show()
    plt.close()
