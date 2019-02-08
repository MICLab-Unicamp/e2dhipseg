'''
Main training/testing functions

Author: Diedre Carmo
https://github.com/dscarmo
'''
import time
import traceback
import os
import glob
import nibabel as nib
import copy
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import cv2 as cv
from utils import ESC, imagePrint, myrotate
from train_results import TrainResults
#from tqdm import tqdm
from transforms import ToNumpy, CenterCrop
from utils import myrotate
from label import get_largest_components
from dataset import FloatHippocampusDataset, ADNI, default_adni
from shutil import copyfile

def train_model(model, criterion, metric, optimizer, scheduler, dataloaders, device, dataset_sizes, 
                inepoch_plot=True, loss_name="Loss", metric_name="Accuracy", model_name="model", num_epochs=10, save=True, savefolder="tests"):
    '''
    Treina um modelo utilizando loss e otimizador passados
    '''
    os.makedirs(savefolder, exist_ok=True)
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    
    try:
        # For each epoch
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)
            loss_per_batch = []

            # Perform trainig and validation
            for phase in ['train', 'validation']:
                if phase == 'train':
                    scheduler.step() 
                    model.train() 
                else:
                    model.eval()

                running_loss = 0.0
                running_acc = 0.0

                # In a EPOCH, iterate over all batches
                for inputs, labels in dataloaders[phase]:
                    batch_size = inputs.size(0)
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    
                    # Use gradients only in training phase, loss and metric should be batch means!
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        acc = metric(outputs, labels)
                        
                        # Backpropagation only in trainig phase
                        if phase == 'train':
                            loss_per_batch.append(loss)
                            loss.backward()
                            optimizer.step()
                            
                    # Batch statistics, multiplies by batch size due to dividing by dataset size later
                    running_loss += loss.item()*batch_size
                    running_acc += acc*batch_size
                    
                # Get epoch mean
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_acc / dataset_sizes[phase]
                
                # Saving stats history
                if phase == 'train':
                    train_loss.append(epoch_loss)
                    train_acc.append(epoch_acc)
                    if inepoch_plot:
                        plt.title("Training Loss per batch, EPOCH: " + str(epoch + 1)) # inicialização de gráfico de loss
                        plt.xlabel('Batch')
                        plt.ylabel(loss_name)
                        plt.plot(range(len(loss_per_batch)), loss_per_batch, 'b*-', label='Train')
                        plt.show()
                elif phase == 'validation':
                    val_loss.append(epoch_loss)
                    val_acc.append(epoch_acc) 
                else:
                    raise ValueError("WRONG PHASE NAME, check folder names?")
                
                print('{} {}: {:.4f} {}: {:.4f} '.format(
                    phase, loss_name, epoch_loss, metric_name, epoch_acc))

                # Keep copy of best validation model
                if phase == 'validation' and epoch_acc > best_acc:
                    print("Best model so far, checkpoint...")
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), os.path.join(savefolder, model_name + ".pt"))
                    best_model_wts = copy.deepcopy(model.state_dict())

    except KeyboardInterrupt as ki:
        print("Training interrupted by ctrl-C")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best accuracy: {:4f}'.format(best_acc))

    # Return best model and result history
    model.load_state_dict(best_model_wts)
    results = TrainResults(val_loss, val_acc,
                           train_loss, train_acc, num_epochs, loss_name, metric_name, plot_title=model_name)
    if save:
        i = 0
        pkl_path = os.path.join(savefolder, model_name + ".pkl")
        while os.path.isfile(pkl_path):
            i += 1
            print(pkl_path + " already exists") 
            pkl_path = os.path.join(savefolder, model_name + str(i) + ".pkl")
        results.save(os.path.join(savefolder, model_name + ".pkl"))
        print("Training data saved to {}".format(pkl_path))

    return results


def test_model(model, criterion, metric, dataloaders, dataset_sizes, device,
               loss_name="Loss", metric_name="Accuracy"):
    '''
    Test the model
    '''
    since = time.time()
    
    print("Running test in test dataloader.")
    model.eval()

    running_loss = 0.0
    running_acc = 0.0

    for inputs, labels in dataloaders["test"]:
        batch_size = inputs.size(0)
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc = metric(outputs, labels)

        # Computa estatisticas do batch
        running_loss += loss.item() * inputs.size(0)
        running_acc += acc * inputs.size(0) 

    # Depois que passa por todos os dados, statisticas do epoch
    test_loss = running_loss / dataset_sizes["test"]
    test_acc = running_acc / dataset_sizes["test"]
    

    print('Test {}: {:.4f} {}: {:.4f} '.format(loss_name, test_loss, metric_name, test_acc))

    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

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
        
def outsider_test(path, name, db, metric):
    '''
    Performs tests with other methods
    '''
    valid_names = ["hippodeep"]
    valid_dbs = ["adni", "clarissa"]
    assert (name in valid_names) and (db in valid_dbs), "{} name or ".format(name)

    accs = []
    if db == "adni":
        db = ADNI()
        volids = FloatHippocampusDataset(h5path=default_adni, return_volume=True, mode="test", adni=True, data_split=(0.0, 0.0, 1)).get_volids()
    elif db == "clarissa":
        db = FloatHippocampusDataset(mode="test", return_volume=True)
        volids = db.get_volids()
        
    for i in volids:
        if i == "33":
            #continue
            pass
        ref = db.get_by_name(str(i))[1]
        mask_L = nib.load(os.path.join(path, str(i) + "_mask_L.nii.gz")).get_fdata()
        mask_R = nib.load(os.path.join(path, str(i) + "_mask_R.nii.gz")).get_fdata()
        mask = mask_L + mask_R
        mask = mask.astype(np.bool).astype(np.float32)
        ref = torch.from_numpy(ref)
        mask_t = torch.from_numpy(mask)
        try:
            if torch.cuda.is_available():
                acc = metric(ref.cuda(), mask_t.cuda())
            else:
                acc = metric(ref, mask_t)
        except Exception as e:
            print("{} failed! {}".format(name,e))
            acc = 0
        print(acc)
        accs.append(acc)

    mean = np.array(accs).mean()
    print(accs)
    print("path: {}, name: {}, db: {}, metric: {}, Mean: {}".format(path, name, db, str(metric), mean))
    return mean
    
def per_volume_test(models, metric, slice_transform, dataset, device, metric_name="Accuracy", display=False, study_ths=False, e2d=False, wait=1, name=""):
    '''
    Test evaluating per volume DICE
    '''
    since = time.time()
    orientations = dataset.reconstruction_orientations

    for o, model in models.items():
        model.eval()
        model.to(device)

    volume_accs = {}
    c_acc = []
    for o in orientations:
        volume_accs[o] = []
    ths_study = [[] for _ in range(9)]
    cons = [[] for _ in range(4)]
    cons_buffer = [0, 0, 0]
    tn = ToNumpy()
    center_p_size = 160
    errors = 0
    abort = False
    for k in range(len(dataset)):
        skip = False
        sample_v, mask_v = dataset[k]
        shape = sample_v.shape
        sum_vol_total = torch.zeros(shape)
        #print("Volume shape: {}".format(shape))
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
                    if e2d == False:
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
                    
                    probs = models[o](s.unsqueeze(0))
                    
                    cpup = probs.squeeze().detach().cpu()
                    volume_o.append(cpup)
                    finalp = torch.from_numpy(myrotate(cpup.numpy(), -90)).float() # back to volume orientation

                    # Add to final consensus volume, uses original orientation
                    if i == 0:
                        toppad = shape[1]//2 - center_p_size//2
                        sidepad = shape[2]//2 - center_p_size//2
                        
                        tf = 1 if shape[1]%2 == 1 else 0
                        sf = 1 if shape[2]%2 == 1 else 0
                        pad = F.pad(finalp, (sidepad + sf, sidepad, toppad, toppad + tf))/3

                        sum_vol_total[j, :, :] += pad
                    elif i == 1:
                        toppad = shape[0]//2 - center_p_size//2
                        sidepad = shape[2]//2 - center_p_size//2

                        tf = 1 if shape[0]%2 == 1 else 0
                        sf = 1 if shape[2]%2 == 1 else 0
                        pad = F.pad(finalp, (sidepad+ sf, sidepad , toppad, toppad + tf))/3

                        sum_vol_total[:, j, :] += pad
                    elif i == 2:
                        toppad = shape[0]//2 - center_p_size//2
                        sidepad = shape[1]//2 - center_p_size//2

                        tf = 1 if shape[0]%2 == 1 else 0
                        sf = 1 if shape[1]%2 == 1 else 0
                        pad = F.pad(finalp, (sidepad + sf, sidepad, toppad, toppad + tf))/3

                        sum_vol_total[:, :, j] += pad
                    

                    if display:
                        if s.shape[0] == 3:
                            display_sample = tn(s[1].cpu())
                        else:
                            display_sample = tn(s.cpu())
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
            volume_o = torch.from_numpy(get_largest_components(volume_o.numpy(), mask_ths=0.5))
            
            o_acc = metric(volume_o.cuda(), volume_m.cuda())
            
            print("{} volume {} orientation {} acc {}".format(metric_name, k, o, o_acc))
            volume_accs[o].append(o_acc)
            cons_buffer[i] = o_acc

        if abort: break
        if skip: continue

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
            #plt.show()

        final_nppred = get_largest_components(sum_vol_total.numpy(), mask_ths=0.5)
        final_vol = torch.from_numpy(final_nppred)

        # mask_compare(sample_v, mask_v, final_nppred) for visual comparison

        f_acc = metric(final_vol.cuda(), final_mask.cuda())
        cons[3].append(f_acc)
        for i in range(3):
            cons[i].append(cons_buffer[i]) 
        c_acc.append(f_acc)
        print("Final consensus acc {}".format(f_acc))
            
    #print("{} per orientation per volume: {}".format(metric_name, str(volume_accs)))
    '''plt.figure(num=name)
    for i, o in enumerate(orientations):
        plt.subplot(1, 4, i + 1)
        plt.title(o)
        plt.ylim(0.0, 1.0)
        plt.plot(range(len(volume_accs[o])), volume_accs[o], 'b*-', label=o)
    plt.subplot(1, 4, 4)
    plt.title("Consensus")
    plt.ylim(0.0, 1.0)
    plt.plot(range(len(c_acc)), c_acc, 'r*-', label="Consensus")'''

    print(c_acc)
    mean_consensus = np.array(c_acc).mean()
    print("Mean consensus DICE: {}".format(mean_consensus))
    print("Number of errors: {}".format(errors))

    return mean_consensus, ths_study, cons

def mask_compare(vol, ref_mask, pred_mask):
    '''
    Performs visual comparison of a reference mask and another mask in a volume
    '''
    shape = vol.shape
    for i in range(shape[0]):
        vols, rmasks, pmasks = vol[i, :, :], ref_mask[i, :, :], pred_mask[i, :, :]
        #vols, rmasks, pmasks = vol[:, i, :], ref_mask[:, i, :], pred_mask[:, i, :]
        #vols, rmasks, pmasks = vol[:, :, i], ref_mask[:, :, i], pred_mask[:, :, i]
        _, rcontour, _ = cv.findContours((rmasks*255).astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        _, pcontour, _ = cv.findContours((pmasks*255).astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        display1 = cv.cvtColor(vols, cv.COLOR_GRAY2RGB)
        display2 = cv.cvtColor(vols, cv.COLOR_GRAY2RGB)

        cv.drawContours(display1[:], rcontour, -1, (1, 0, 0), 1)
        cv.drawContours(display2, pcontour, -1, (0, 1, 0), 1)
        
        #sagital
        display1 = display1[50:150, 70:170]
        display2 = display2[50:150, 70:170]

        # axial
        #display1 = display1[75:175, 50:150]
        #display2 = display2[75:175, 50:150]

        prerot = np.hstack([display1, display2])

        display1 = myrotate(display1, 90)
        display2 = myrotate(display2, 90)

        display = np.hstack([display1, display2])

        cv.imshow("prerot", (prerot*255).astype(np.uint8))
        cv.imshow("hippodeep - me", (display*255).astype(np.uint8))
        if cv.waitKey(0) == 27:
            cv.imwrite("display.png", (display*255).astype(np.uint8))
            quit()

if __name__ == "__main__":
    print("Rodando teste em ADNI usando hippodeep")
    from metrics import DICEMetric
    from sys import argv
    
    metric = DICEMetric(apply_sigmoid=False)
    try:
        arg = argv[1]
    except:
        print("Please give arg: which dataset to run hippodeep")

    if arg == "adni":
        outsider_test("/home/diedre/bigdata/manual_selection_rotated/hippodeep", "hippodeep", "adni", DICEMetric(apply_sigmoid=False))
    elif arg == "clarissa":
        outsider_test("/home/diedre/bigdata/mni_hip_data/hippodeep", "hippodeep", "clarissa", DICEMetric(apply_sigmoid=False))
    else:
        raise ValueError("Didnt recognize arg {}".format(arg))