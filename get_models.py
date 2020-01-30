'''
Abstracts getting model objects
'''
import glob
import os

from dataset import orientations
import torch
from unet import UNet
from dunet import get_dunet


def get_models(bias, e2d, res, small, bn, dunet, dim='2d', model_folder=None, verbose=True, out_channels=1, apply_sigmoid=True,
               classify=False, apply_softmax=False):
    '''
    Navigates through past results folder to load a past result
    '''
    if dim == '3d':
        model = UNet(1 + (e2d*2), out_channels, residual=res, small=small, bias=bias, bn=bn, dim=dim, verbose=verbose,
                     apply_sigmoid=apply_sigmoid, classify_ad=classify, use_attention=classify)
        if model_folder is not None:
            path = glob.glob(os.path.join(model_folder, "*.pt"))[0]
            model.load_state_dict(torch.load(path, map_location="cpu" if not torch.cuda.is_available() else None))
        return model
    else:
        models = {}
        for o in orientations:
            if dunet:
                model = get_dunet()
            else:
                model = UNet(1 + (e2d*2), out_channels, residual=res, small=small, bias=bias, bn=bn, dim=dim, verbose=verbose,
                             apply_sigmoid=apply_sigmoid, classify_ad=classify, use_attention=classify,
                             apply_softmax=apply_softmax)
            if model_folder is not None:
                print("Getting weights from {}".format(model_folder))
                path = glob.glob(os.path.join(model_folder, "*" + o + ".pt"))[0]
                model.load_state_dict(torch.load(path, map_location="cpu" if not torch.cuda.is_available() else None))
            models[o] = model
        return models


def get_dti_model(inchannels, bias, res, small, bn, dunet, dim='2d', model_folder=None, verbose=True, out_channels=1,
                  apply_sigmoid=True):
    '''
    For DTI seg
    Navigates through past results folder to load a past result
    '''
    model = UNet(inchannels, out_channels, residual=res, small=small, bias=bias, bn=bn, dim=dim, verbose=verbose,
                 apply_sigmoid=apply_sigmoid)

    if model_folder is not None:
        print("Getting weights from {}".format(model_folder))
        path = glob.glob(os.path.join(model_folder, "*.pt"))[0]
        model.load_state_dict(torch.load(path))

    return model
