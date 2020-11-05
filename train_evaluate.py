'''
Training/testing functions.

Author: Diedre Carmo
https://github.com/dscarmo
'''
import time
import traceback
import collections
import os
import nibabel as nib
import copy
from matplotlib import pyplot as plt
from glob import glob
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import cv2 as cv
from utils import (ESC, imagePrint, myrotate, viewnii, show_multichannel_slice, half_multi_task_labels, multi_task_labels,
                   HALF_MULTI_TASK_NCHANNELS, MULTI_TASK_NCHANNELS, ce_output_to_mask, cv_display_attention, probs2one_hot,
                   GDL_TWO)
from train_results import TrainResults
from transforms import ToNumpy, CenterCrop, perform_random_rotation
from label import get_largest_components
from dataset import FloatHippocampusDataset, ADNI, default_adni, mni_adni, default_harp, HARP_CLASSES
from dataset import orientations as default_orientations
from radam import RAdam
from nathip import NatHIP
from metrics import EvalMetricCalculator


def get_optimizer_scheduler(model, args, **kwargs):
    '''
    Returns optimizer, scheduler and other training parameters
    '''
    if args["adam"]:
        print("Using adam with LR {}".format(kwargs["LR"]))
        opt = optim.Adam(model.parameters(), lr=kwargs["LR"])
    elif args["radam"]:
        print("Using RADAM with LR: {}".format(kwargs["LR"]))
        opt = RAdam(model.parameters(), lr=kwargs["LR"])
    else:
        opt = optim.SGD(model.parameters(), lr=kwargs["LR"], momentum=0.9)

    if args["NOSTEP"]:
        print("Not using scheduler step_size")
        step_size = kwargs["NEPOCHS"]
    elif args["newschedule"]:
        step_size = kwargs["patience"]
    else:
        step_size = kwargs["NEPOCHS"]//2
        if args["QSTEP"]:
            print("QSTEP arg found, halfing stepsize")
            step_size = step_size//2
        elif args["4QSTEP"]:
            print("4QSTEP arg found, 1/4 stepsize")
            step_size = step_size//4
        if step_size < 1:
            step_size = 1

    gamma = 0.9 if args["newschedule"] else 0.1
    scheduler = lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

    print("Scheduler step size: {}, gamma: {}".format(step_size, gamma))

    return opt, scheduler


def train_model(model, criterion, metric, optimizer, scheduler, dataloaders, device, dataset_sizes,
                inepoch_plot=True, loss_name="Loss", metric_name="Accuracy", model_name="model",
                num_epochs=10, save=True, savefolder="tests", patience=0, classify=False, probs2onehot=False):
    '''
    Treina um modelo utilizando loss e otimizador passados
    '''
    os.makedirs(savefolder, exist_ok=True)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0
    patience_count = 0
    val_loss = []
    val_acc = []

    train_loss = []
    train_acc = []
    if classify:
        val_class = []
        train_class = []
    else:
        val_class = None
        train_class = None
    boundary_loss = hasattr(criterion, "max_ncalls")
    try:
        # For each epoch
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)
            loss_per_batch = []

            # Perform trainig and validation
            for phase in ['train', 'validation']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                running_loss = 0.0
                running_acc = 0.0
                running_class = 0.0

                # In a EPOCH, iterate over all batches
                for inputs, labels in dataloaders[phase]:
                    batch_size = inputs.size(0)

                    inputs = inputs.to(device)

                    if isinstance(labels, collections.Sequence):
                        if boundary_loss:
                            dists = labels[1].to(device)
                            labels = labels[0].to(device)
                        else:
                            labels = (labels[0].to(device), labels[1].to(device))
                    else:
                        labels = labels.to(device)

                    optimizer.zero_grad()

                    # Use gradients only in training phase, loss and metric should be batch means!
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)

                        if boundary_loss:
                            loss = criterion(outputs, dists, labels)
                        else:
                            loss = criterion(outputs, labels)

                        if probs2onehot:
                            mask = probs2one_hot(outputs.detach())
                        else:
                            mask = outputs

                        if hasattr(criterion, "cross_entropy") and criterion.cross_entropy:
                            outputs = ce_output_to_mask(outputs, numpy=False).to(device)

                        if classify:
                            acc, acc_class = metric(outputs, labels)
                        else:
                            acc = metric(mask, labels)

                        # Backpropagation only in trainig phase
                        if phase == 'train':
                            loss_per_batch.append(loss)
                            loss.backward()
                            optimizer.step()

                    # Batch statistics, multiplies by batch size due to dividing by dataset size later
                    running_loss += loss.item()*batch_size
                    running_acc += acc*batch_size
                    if classify:
                        running_class += acc_class*batch_size

                if phase == "train":
                    print("Learning rate step")
                    scheduler.step()

                # Get epoch mean
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_acc / dataset_sizes[phase]
                epoch_class = running_class / dataset_sizes[phase]

                # Post epoch computation tasks
                if phase == 'train':
                    # If using boundary loss, increment weights
                    if boundary_loss:
                        criterion.increment_weights()
                    train_loss.append(epoch_loss)
                    train_acc.append(epoch_acc)
                    if classify:
                        train_class.append(epoch_class)
                    # DEPRECATED
                    if inepoch_plot:
                        plt.title("Training Loss per batch, EPOCH: " + str(epoch + 1))  # inicialização de gráfico de loss
                        plt.xlabel('Batch')
                        plt.ylabel(loss_name)
                        plt.plot(range(len(loss_per_batch)), loss_per_batch, 'b*-', label='Train')
                        plt.show()
                elif phase == 'validation':
                    val_loss.append(epoch_loss)
                    val_acc.append(epoch_acc)
                    if classify:
                        val_class.append(epoch_class)
                else:
                    raise ValueError("WRONG PHASE NAME, check folder names?")

                print('{} {}: {:.4f} {}: {:.4f} '.format(phase, loss_name, epoch_loss, metric_name, epoch_acc) +
                      classify*"Class ACC: {:.4f}".format(epoch_class))

                # Keep copy of best validation model
                if phase == 'validation':
                    if epoch_acc > best_acc or epoch_class > best_acc:
                        if patience != 0:
                            patience_count = 0

                        print("Best model so far, checkpoint...")
                        best_acc = epoch_acc if epoch_acc > best_acc else epoch_class
                        torch.save(model.state_dict(), os.path.join(savefolder, model_name + ".pt"))
                        best_model_wts = copy.deepcopy(model.state_dict())
                    elif patience != 0:
                        patience_count += 1
                        print("Patience {}/{}...".format(patience_count, patience))
                        if patience_count >= patience:
                            print("{} epochs without improvement, ending training.".format(patience_count))
                            raise KeyboardInterrupt("Patience interrupt")

    except KeyboardInterrupt as ki:
        print("Training interrupted: {}".format(ki))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best accuracy: {:4f}'.format(best_acc))

    # Return best model and result history
    model.load_state_dict(best_model_wts)
    results = TrainResults(val_loss, val_acc, train_loss, train_acc, num_epochs, loss_name, metric_name, plot_title=model_name,
                           train_class=train_class, val_class=val_class)
    if save:
        i = 0
        pkl_path = os.path.join(savefolder, model_name + ".pkl")

        with open(os.path.join(savefolder, "final_val_acc{}.txt".format(best_acc)), 'w') as print_acc:
            print_acc.write(str(best_acc))

        while os.path.isfile(pkl_path):
            i += 1
            print(pkl_path + " already exists")
            pkl_path = os.path.join(savefolder, model_name + str(i) + ".pkl")
        results.save(pkl_path)
        print("Training data saved to {}".format(pkl_path))

    return results


def test_model(model, criterion, metric, dataloaders, dataset_sizes, device,
               loss_name="Loss", metric_name="Accuracy", filter_components=False, display_sample=False, save_path=None, db=None,
               classify=False, probs2onehot=False):
    '''
    Test the model
    '''
    metricer = EvalMetricCalculator(metric)
    since = time.time()
    print("Running test in test dataloader.")
    model.eval()

    running_loss = 0.0
    running_acc = 0.0
    running_class = 0.0
    multi_channel = False
    acc_history = []

    acc_class = 0
    boundary_loss = hasattr(criterion, "max_ncalls")
    for inputs, labels in dataloaders["test"]:
        batch_size = inputs.size(0)

        inputs = inputs.to(device)

        if isinstance(labels, collections.Sequence):
            if boundary_loss:
                dists = labels[1].to(device)
                labels = labels[0].to(device)
            else:
                labels = (labels[0].to(device), labels[1].to(device))
        else:
            labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            if not isinstance(outputs, collections.Sequence) and not classify:
                if outputs.shape[1] != 1:
                    multi_channel = True

            if boundary_loss:
                loss = criterion(outputs, dists, labels)
            else:
                loss = criterion(outputs, labels)

            if probs2onehot:
                mask = probs2one_hot(outputs.detach())
            else:
                mask = outputs

            if filter_components:  # filtering output because testing 3D unet
                assert not classify, "classify not support in volumetric yet"
                assert batch_size == 1, "volumetric testing should have batch size equal to 1"
                npinput, npoutput, ref = (inputs[0, 0, :, :, :].cpu().numpy(), mask[0, 0, :, :, :].detach().cpu().numpy(),
                                          labels[0, :, :, :].cpu().numpy())
                if display_sample:
                    pre_acc = metric(mask, labels)
                    viewnii(npinput, npoutput, ref=ref, id="Before pre processing DICE {}".format(pre_acc), wait=1,
                            border_only=False)
                larg_comp = get_largest_components(npoutput, mask_ths=0.5)
                preds = torch.from_numpy(larg_comp).to(device)
                acc = metric(preds, labels)
                metricer(preds, labels.squeeze())
                if display_sample:
                    viewnii(npinput, larg_comp, ref=ref, id="DICE: {}".format(acc), wait=1, border_only=False)
            elif multi_channel:  # multitask
                assert not classify, "classify not support in multitask yet"
                if hasattr(criterion, "cross_entropy") and criterion.cross_entropy:
                    outputs = ce_output_to_mask(outputs, numpy=False)

                if display_sample:
                    if outputs.shape[1] == HALF_MULTI_TASK_NCHANNELS:
                        display_labels = half_multi_task_labels
                    else:
                        display_labels = multi_task_labels

                    if hasattr(criterion, "cross_entropy") and criterion.cross_entropy:
                        npoutput = mask.numpy()
                    else:
                        npoutput = mask.detach().cpu().numpy()

                    npinput, npref = inputs.cpu().numpy(), labels.cpu().numpy()
                    show_multichannel_slice(npinput, npoutput, npref, multichannel_labels=display_labels)
                acc = metric(mask.to(device), labels)
            else:
                if classify:
                    acc, acc_class = metric(mask, labels)
                else:
                    acc = metric(mask, labels)
                    metricer(mask, labels)
        # Computa estatisticas do batch
        running_loss += loss.item() * inputs.size(0)
        running_acc += acc * inputs.size(0)
        running_class += acc_class*inputs.size(0)
        acc_history.append(acc)

    # Depois que passa por todos os dados, statisticas do epoch
    test_loss = running_loss / dataset_sizes["test"]
    test_acc = running_acc / dataset_sizes["test"]
    test_class = running_class / dataset_sizes["test"]

    print('Test {}: {:.4f} {}: {:.4f}'.format(loss_name, test_loss, metric_name, test_acc) +
          classify*" Class acc: {:.4f}".format(test_class))
    with open(os.path.join(save_path, str(test_acc) + '_' + classify*str(test_class) + str(db) + "_test.txt"), 'a') as f:
        f.write(str(test_acc) + '\n' + str(test_class))

    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s for {} volumes/slices'.format(
        time_elapsed // 60, time_elapsed % 60, dataset_sizes["test"]))

    acc_history = np.array(acc_history)
    print("Mean: {}, std: {}".format(acc_history.mean(), acc_history.std()))

    return metricer.final_results()


def display_test(s, m, p, wait=1):
    '''
    Shows slices and predictions during volume testing
    '''
    size = (160, 160)
    view1 = cv.resize(s, size)
    view2 = cv.resize(m, size)
    view3 = (view1 + view2)/2
    view4 = cv.resize(p, size)
    view5 = (view1 + view4)/2

    imagePrint(view1, "sample")
    imagePrint(view2, "mask")
    imagePrint(view3, "s + m")
    imagePrint(view4, "heatmap")
    imagePrint(view5, "s + h")

    view = np.hstack((view1, view2, view3, view4, view5))
    view = cv.resize(view, (0, 0), fx=2, fy=2)
    cv.imshow("Volume testing...", view)
    return cv.waitKey(wait)


def outsider_test(path, name, db, metric, display=False, group="all"):
    '''
    Performs tests with other methods
    '''
    valid_names = ["hippodeep"]
    valid_dbs = ["adni", "mnihip", "mniadni", "harp", "nathip"]
    assert (name in valid_names) and (db in valid_dbs), "{} name or ".format(name)

    calculator = EvalMetricCalculator(metric)

    if db == "adni":
        db = ADNI(mnireg=False)
        volids = FloatHippocampusDataset(h5path=default_adni, return_volume=True, mode="test", adni=True,
                                         data_split=(0.0, 0.0, 1), mnireg=False).get_volids()
    elif db == "mniadni":
        db = ADNI(mnireg=True)
        volids = FloatHippocampusDataset(h5path=mni_adni, return_volume=True, mode="test", adni=True, data_split=(0.0, 0.0, 1),
                                         mnireg=True).get_volids()
    elif db == "mnihip":
        db = FloatHippocampusDataset(mode="test", return_volume=True)
        volids = db.get_volids()
    elif db == "harp":
        db = FloatHippocampusDataset(h5path=default_harp, return_volume=True, mode="test", adni=False, harp=True,
                                     data_split=(0.0, 0.0, 1), mnireg=False)
        volids = db.get_volids()
    elif db == "nathip":
        db = NatHIP(group=group, mode="all", fold=None, orientation=None, transform=None, e2d=False, return_onehot=False,
                    verbose=True)
        volids = db.get_volids()

    for i in volids:
        try:
            print("DICE for {}".format(i))
            vol, ref = db.get_by_name(str(i))
            mask_L = nib.load(glob(os.path.join(path, "*" + str(i) + "_mask_L.nii.gz"))[0]).get_fdata()
            mask_R = nib.load(glob(os.path.join(path, "*" + str(i) + "_mask_R.nii.gz"))[0]).get_fdata()
        except FileNotFoundError as fnfe:
            print("File not found! {}".format(fnfe))

        mask = mask_L + mask_R
        mask = mask.astype(np.bool).astype(np.float32)
        if display:
            print("displaying")
            viewnii(vol, mask=mask, ref=ref, wait=0, rotate=0, quit_on_esc=True, border_only=False)
        ref = torch.from_numpy(ref)
        mask_t = torch.from_numpy(mask)
        try:
            if torch.cuda.is_available():
                calculator(mask_t.cuda(), ref.cuda())
            else:
                calculator(mask_t, ref)
        except Exception as e:
            print("{} failed! {}".format(name, e))

        print("DICE: {}, Moving Mean: {}, Moving Standard deviation: {}".format(calculator.acc, np.array(calculator.accs).mean(),
                                                                                np.array(calculator.accs).std()))

    print("Path: {}, name: {}, db: {}, metric: {})".format(path, name, db, str(metric)))
    calculator.final_results()


def pad_and_consensus(oshape, croped_slice, sum_volume, orientation, slice_position, patch_size, weight=3):
    '''
    oshape: shape that you intend to pad the croped volume to
    croped_volume: the smaller volume
    sum_volume: where the paded slice will go, divided by weight
    weight: factor to apply to values in slice
    i: orientation
    Pad a slice back to a intended shape and add it to a to final sum volume
    '''

    if orientation == 0:
        toppad = oshape[1]//2 - patch_size//2
        sidepad = oshape[2]//2 - patch_size//2

        tf = 1 if oshape[1] % 2 == 1 else 0
        sf = 1 if oshape[2] % 2 == 1 else 0
        pad = F.pad(croped_slice, (sidepad + sf, sidepad, toppad, toppad + tf))/weight

        sum_volume[slice_position, :, :] += pad
    elif orientation == 1:
        toppad = oshape[0]//2 - patch_size//2
        sidepad = oshape[2]//2 - patch_size//2

        tf = 1 if oshape[0] % 2 == 1 else 0
        sf = 1 if oshape[2] % 2 == 1 else 0
        pad = F.pad(croped_slice, (sidepad + sf, sidepad, toppad, toppad + tf))/weight

        sum_volume[:, slice_position, :] += pad
    elif orientation == 2:
        toppad = oshape[0]//2 - patch_size//2
        sidepad = oshape[1]//2 - patch_size//2

        tf = 1 if oshape[0] % 2 == 1 else 0
        sf = 1 if oshape[1] % 2 == 1 else 0
        pad = F.pad(croped_slice, (sidepad + sf, sidepad, toppad, toppad + tf))/weight

        sum_volume[:, :, slice_position] += pad


def per_volume_test(models, metric, slice_transform, dataset, device, metric_name="Accuracy", display=False, study_ths=False,
                    e2d=False, wait=1, name="", ce_output=False, classify=False, rot=False):
    '''
    Test evaluating per volume DICE
    '''
    print("Output is crossentropy? {}, Using E2D input? {}".format(ce_output, e2d))
    for o, model in models.items():
        model.eval()
        model.to(device)

    metricer = EvalMetricCalculator(metric)
    volume_accs = {}
    c_acc = []
    nc_acc = []
    for o in default_orientations:
        volume_accs[o] = []
    ths_study = [[] for _ in range(9)]
    cons = [[] for _ in range(8)]  # hold accs for each orientation with post processing and consensus
    tn = ToNumpy()
    center_p_size = 160
    errors = 0
    abort = False
    ncorrect = 0
    nprocessed = 0
    for k in range(len(dataset)):
        skip = False
        label_acum = torch.tensor([0.0, 0.0, 0.0])
        if classify:
            sample_v, (mask_v, target_v) = dataset[k]
            # att_vol = np.zeros(sample_v.shape)  # shape inconsistency problem
        else:
            sample_v, mask_v = dataset[k]

        if rot:
            print("WARNING: Performing random rotations for testing purposes")
            sample_v, mask_v = perform_random_rotation(sample_v, tgt=mask_v)
            print(sample_v.shape, mask_v.shape)

        shape = sample_v.shape
        sum_vol_total = torch.zeros(shape)
        # print("Volume shape: {}".format(shape))

        orientations = default_orientations
        print("Assuming volume is in {}".format(orientations))

        for i, o in enumerate(orientations):
            if display:
                print("Volume Testing {}".format(o))
            volume_m = []
            volume_o = []
            try:
                for j in range(shape[i]):
                    # Get central mask
                    if i == 0:
                        tm = mask_v[j, :, :]
                    elif i == 1:
                        tm = mask_v[:, j, :]
                    elif i == 2:
                        tm = mask_v[:, :, j]
                    tm = myrotate(tm, 90)

                    # Get 3 slices if e2d true
                    if e2d is False:
                        if i == 0:
                            ts = sample_v[j, :, :]
                        elif i == 1:
                            ts = sample_v[:, j, :]
                        elif i == 2:
                            ts = sample_v[:, :, j]
                        ts = myrotate(ts, 90)
                    else:
                        ts = np.zeros((3, tm.shape[0], tm.shape[1]), dtype=tm.dtype)
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

                    s, m = slice_transform(ts, tm)
                    volume_m.append(m)
                    s = s.to(device)
                    m = m.to(device)

                    with torch.set_grad_enabled(False):
                        if classify:  # DEPRECATED
                            probs, label, att = models[o](s.unsqueeze(0))  # unsqueeze makes the data represent a batch of 1
                            att = att.detach().squeeze().cpu().numpy()
                            if m.sum() > 0:  # Only classify in label presence
                                label_acum += label.detach().cpu().squeeze()
                        else:
                            probs = models[o](s.unsqueeze(0))  # unsqueeze makes the data represent a batch of 1

                    if ce_output:
                        probs = ce_output_to_mask(probs, numpy=False).to(device)

                    # Multitask workaround
                    nchannels = probs[0].shape[1] if classify else probs.shape[1]
                    if nchannels == MULTI_TASK_NCHANNELS:
                        probs = probs[:, 11, :, :] + probs[:, 12, :, :]
                    elif nchannels == HALF_MULTI_TASK_NCHANNELS:
                        if display:
                            for debut_i in range(9):
                                cv.imshow("channel {}".format(debut_i),
                                          np.hstack((m[0].cpu().numpy(),
                                                     probs[:, debut_i, :, :].squeeze().detach().cpu().numpy())))
                        probs = probs[:, 6, :, :]
                    elif nchannels == GDL_TWO:
                        probs = probs[:, 1, :, :]

                    cpup = probs.squeeze().detach().cpu()
                    volume_o.append(cpup)
                    finalp = torch.from_numpy(myrotate(cpup.numpy(), -90)).float()  # back to volume orientation

                    pad_and_consensus(shape, finalp, sum_vol_total, i, j, center_p_size, weight=3)

                    if display:
                        if s.shape[0] == 3:
                            display_sample = tn(s[1].cpu())
                        else:
                            display_sample = tn(s.cpu())
                        if classify:
                            cv_display_attention(att, display_engine='cv')
                        if display_test(display_sample, tn(m.cpu()), tn(cpup), wait=wait) == ESC:
                            print("Ending...")
                            abort = True
                            break
            except Exception as e:
                print("Error testing volume: {}".format(e))
                traceback.print_exc()
                errors += 1
                skip = True

            if skip or abort:
                break

            volume_m = torch.stack(volume_m).squeeze(1)
            volume_o = torch.stack(volume_o)
            nop_acc = metric(volume_o.to(device), volume_m.to(device))
            volume_o = torch.from_numpy(get_largest_components(volume_o.numpy(), mask_ths=0.5))

            o_acc = metric(volume_o.to(device), volume_m.to(device))

            print("{} volume {} orientation {} acc {} nop_acc {}".format(metric_name, k, o, o_acc, nop_acc))
            volume_accs[o].append(o_acc)
            cons[2*i].append(nop_acc)  # nops, os, nopc, oc, nopa, oa
            cons[2*i + 1].append(o_acc)

        if abort:
            quit()

        if skip:
            continue

        # computing consensus accuracy
        final_mask = torch.from_numpy(mask_v).cuda()
        if study_ths:
            accs = []
            for r in range(1, 10):
                print(r/10)
                final_vol = torch.from_numpy(get_largest_components(sum_vol_total.numpy(), mask_ths=r/10))
                ths_acc = metric(final_vol.cuda(), final_mask)
                ths_study[r-1].append(ths_acc)
                accs.append(ths_acc)
            plt.figure()
            plt.plot(range(len(accs)), accs, 'b*-')
            # plt.show()

        nop_facc = metric(sum_vol_total.cuda(), final_mask.cuda())  # consensus acc without post processing
        final_nppred = get_largest_components(sum_vol_total.numpy(), mask_ths=0.5)
        final_vol = torch.from_numpy(final_nppred)
        if display:
            print("displaying final result")
            viewnii(sample_v, mask=final_nppred, ref=mask_v)
            # if classify:
            #     viewnii(att_vol)
        # mask_compare(sample_v, mask_v, final_nppred) for visual comparison

        f_acc = metric(final_vol.cuda(), final_mask.cuda())  # consensus acc with post processing
        metricer(final_vol.cuda(), final_mask.cuda())

        # Confirmed harp has wrong gold standard masks
        # if f_acc < 0.7:
        #     viewnii(sample_v, mask=final_nppred, ref=mask_v, wait=0, border_only=False)
        cons[6].append(nop_facc)
        cons[7].append(f_acc)
        c_acc.append(f_acc)
        nc_acc.append(nop_facc)
        print("No Post Final consensus acc {}".format(nop_facc))
        print("Final consensus acc {}".format(f_acc))

        if classify:
            nprocessed += 1
            label = label_acum.softmax(dim=-1).argmax().item()
            print("Tensor sum: {} out label: {} / {}, target: {} / {}".format(label_acum, label, HARP_CLASSES[label],
                                                                              target_v, HARP_CLASSES[target_v]))
            if target_v == label:
                ncorrect += 1

    print(c_acc)
    std_consensus = np.array(c_acc).std()
    mean_consensus = np.array(c_acc).mean()
    nop_mean_consensus = np.array(nc_acc).mean()
    print("No post mean consensus DICE: {}".format(nop_mean_consensus))
    print("Mean consensus DICE: {}".format(mean_consensus))

    print("Number of errors: {}".format(errors))
    if classify:
        print("Classification Accuracy: {}".format(ncorrect/nprocessed))

    results = metricer.final_results()

    return mean_consensus, ths_study, cons, std_consensus, results


def mask_compare(vol, ref_mask, pred_mask):
    '''
    Performs visual comparison of a reference mask and another mask in a volume
    '''
    shape = vol.shape
    for i in range(shape[0]):
        vols, rmasks, pmasks = vol[i, :, :], ref_mask[i, :, :], pred_mask[i, :, :]
        # vols, rmasks, pmasks = vol[:, i, :], ref_mask[:, i, :], pred_mask[:, i, :]
        # vols, rmasks, pmasks = vol[:, :, i], ref_mask[:, :, i], pred_mask[:, :, i]
        _, rcontour, _ = cv.findContours((rmasks*255).astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        _, pcontour, _ = cv.findContours((pmasks*255).astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        display1 = cv.cvtColor(vols, cv.COLOR_GRAY2RGB)
        display2 = cv.cvtColor(vols, cv.COLOR_GRAY2RGB)

        cv.drawContours(display1[:], rcontour, -1, (1, 0, 0), 1)
        cv.drawContours(display2, pcontour, -1, (0, 1, 0), 1)

        # sagital
        display1 = display1[50:150, 70:170]
        display2 = display2[50:150, 70:170]

        # axial
        # display1 = display1[75:175, 50:150]
        # display2 = display2[75:175, 50:150]

        prerot = np.hstack([display1, display2])

        display1 = myrotate(display1, 90)
        display2 = myrotate(display2, 90)

        display = np.hstack([display1, display2])

        cv.imshow("prerot", (prerot*255).astype(np.uint8))
        cv.imshow("hippodeep - me", (display*255).astype(np.uint8))
        if cv.waitKey(0) == 27:
            cv.imwrite("display.png", (display*255).astype(np.uint8))
            quit()


def dti_volumetric_test(model, metric, dataset, slice_axis, device, metric_name="Accuracy", display=False, wait=1,
                        name="", savefolder="tests", zero_out_smalls=False, display_engine="cv", displasia=False,
                        t1_separate=False):
    '''
    Test evaluating per volume DICE
    '''
    if display_engine == "plt":
        from IPython.display import clear_output

    print("Warning: Zeroing out small structures predictions for correct DICE")
    model.eval()
    model.to(device)

    volume_accs = []

    volume_croper = CenterCrop(128, 128, 128)

    errors = 0
    abort = False
    for k in range(len(dataset)):
        skip = False
        fid, sample_v, mask_v = dataset[k]
        if t1_separate:
            t1 = copy.deepcopy(sample_v[4])
            sample_v = sample_v[:4]
            t1, _ = volume_croper(t1, t1)
        else:
            t1 = None

        crop_sample, crop_mask = volume_croper(sample_v, mask_v)
        crop_output = torch.zeros(crop_mask.shape)
        shape = crop_sample.shape

        try:
            for j in range(shape[slice_axis + 1]):
                # Get central mask
                if slice_axis == 0:
                    sample = crop_sample[:, j, :, :]
                    mask = crop_mask[:, j, :, :]
                    if t1 is not None:
                        t1slice = t1[j, :, :]
                elif slice_axis == 1:
                    sample = crop_sample[:, :, j, :]
                    mask = crop_mask[:, :, j, :]
                    if t1 is not None:
                        t1slice = t1[:, j, :]
                elif slice_axis == 2:
                    sample = crop_sample[:, :, :, j]
                    mask = crop_mask[:, :, :, j]
                    if t1 is not None:
                        t1slice = t1[:, :, j]

                # Center crop
                t_sample, t_mask = torch.from_numpy(sample), torch.from_numpy(mask)

                t_sample = t_sample.to(device)
                t_mask = t_mask.to(device)

                probs = model(t_sample.unsqueeze(0))  # unsqueeze makes the data represent a batch of 1
                probs = probs.squeeze().detach().cpu()

                if zero_out_smalls:
                    probs[0] = 0
                    probs[5:9] = 0

                if slice_axis == 0:
                    crop_output[:, j, :, :] = probs
                elif slice_axis == 1:
                    crop_output[:, :, j, :] = probs
                elif slice_axis == 2:
                    crop_output[:, :, :, j] = probs

                display_probs = probs.numpy()
                if display:
                    if displasia:
                        in_channels = (sample[c] for c in range(sample.shape[0]))
                        target_channels = (mask[c] for c in range(1, mask.shape[0]))
                        out_channels = (display_probs[c] for c in range(1, display_probs.shape[0]))
                    else:
                        in_channels = (sample[c] for c in range(sample.shape[0]))
                        target_channels = list((mask[c] for c in range(1, 5)))
                        out_channels = (display_probs[c] for c in range(1, 5))
                        for en, tc in enumerate(target_channels):
                            sm = tc + t1slice/2
                            sm[sm > 1] = 1
                            target_channels[en] = sm

                    if display_engine == "cv":
                        cv.imshow("input", np.hstack(in_channels))
                        cv.imshow("output", np.hstack(out_channels))
                        cv.imshow("target", np.hstack(target_channels))
                        if cv.waitKey(wait) == 27:
                            print("Interrupting early due to ESC, nothing was saved.")
                            cv.destroyAllWindows()
                            return
                    elif display_engine == "plt":
                        if displasia:
                            (in_channels, target_channels, out_channels) = (list(in_channels), list(target_channels),
                                                                            list(out_channels))
                            in_channels = [in_channels[0], in_channels[1], in_channels[2]]
                            target_channels = [target_channels[0], np.zeros_like(target_channels[0]),
                                               np.zeros_like(target_channels[0])]
                            out_channels = [out_channels[0], np.zeros_like(out_channels[0]), np.zeros_like(out_channels[0])]

                        target_channels = np.hstack(target_channels)
                        out_channels = np.hstack(out_channels)
                        in_channels = np.hstack(in_channels)

                        display_image = np.vstack((in_channels,
                                                   target_channels,
                                                   out_channels))
                        plt.imshow(display_image, cmap='gray')
                        plt.show()
                        time.sleep(wait/1000)
                        clear_output(wait=True)

            acc = metric(torch.from_numpy(crop_mask).unsqueeze(0), crop_output.unsqueeze(0))
            with open(os.path.join(savefolder, str(dataset.folder_name) + "acc_per_volume.csv"), 'a+') as print_acc:
                print_acc.write("{}, {}\n".format(fid, acc))

            volume_accs.append(acc)
            nib.save(nib.Nifti1Image(crop_output[1].detach().cpu().numpy(), affine=None),
                     os.path.join("results", "{}_output.nii.gz".format(fid)))

        except Exception as e:
            print("Error testing volume: {}".format(e))
            traceback.print_exc()
            errors += 1
            skip = True

        if skip or abort:
            break

        if display_engine != "plt":
            print("volume {} {} {}".format(metric_name, k, acc))

        if abort:
            quit()

        if skip:
            continue

    accs_array = np.array(volume_accs)
    mean_dice = accs_array.mean()
    print("Mean DICE: {}".format(mean_dice))
    print("Number of errors: {}".format(errors))

    write_acc_path = os.path.join(savefolder, str(dataset.folder_name) + "test_acc{}_errors{}.txt".format(mean_dice, errors))
    with open(write_acc_path, 'w') as print_acc:
        print_acc.write("Mean DICE: {}\nNumber of errors: {}".format(mean_dice, errors))

    if display:
        cv.destroyAllWindows()

    if metric.per_channel_metric:
        return accs_array
    else:
        plt.plot(accs_array)
        plt.show()

        return mean_dice


if __name__ == "__main__":
    print("Rodando teste em ADNI usando hippodeep")
    from metrics import DICEMetric
    from sys import argv

    display = "display" in argv
    metric = DICEMetric(apply_sigmoid=False)
    try:
        arg = argv[1]
    except IndexError:
        print("Please give arg: which dataset to run hippodeep")
        quit()

    if arg == "adni":
        outsider_test("/home/diedre/Dropbox/bigdata/manual_selection_rotated/hippodeep", "hippodeep", "adni",
                      DICEMetric(apply_sigmoid=False), display=display)
    elif arg == "mniadni":
        outsider_test("/home/diedre/Dropbox/bigdata/manual_selection_rotated/mnihippodeep", "hippodeep", "mniadni",
                      DICEMetric(apply_sigmoid=False), display=display)
    elif arg == "mnihip":
        outsider_test("/home/diedre/Dropbox/bigdata/mni_hip_data/hippodeep", "hippodeep", "mnihip",
                      DICEMetric(apply_sigmoid=False), display=display)
    elif arg == "harp":
        outsider_test("/home/diedre/Dropbox/bigdata/harp/hippodeep", "hippodeep", "harp",
                      DICEMetric(apply_sigmoid=False), display=display)
    elif arg == "nathip":
        group = "all"
        if '-p' in argv:
            group = "PACIENTES"
        elif '-c' in argv:
            group = "CONTROLES"

        outsider_test("/home/diedre/Dropbox/bigdata/nathip/hippodeep", "hippodeep", "nathip",
                      DICEMetric(apply_sigmoid=False), display=display, group=group)
    else:
        raise ValueError("Didnt recognize arg {}".format(arg))
