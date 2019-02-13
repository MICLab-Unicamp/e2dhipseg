# e2dhipseg

# Introduction
This contains official implementation for Extended 2D Volumetric Consensus Hippocampus Segmentation (https://arxiv.org/abs/1902.04487)

We made this version of the code public for it to be available for reviewers/chairs with our current MIDL2019 submission: 
Authors: Diedre Carmo, Bruna Silva, Clarissa Yasuda, Leticia Rittner, Roberto Lotufo
Title: Extended 2D Volumetric Consensus Hippocampus Segmentation

We plan to implement quality of life features for external use soon. Currently most configuration options are arguments or hardcoded. Documentation is on the way.

Improvements are also still being made privately.

Thank you!

# Requirements
python3\
pytorch >= 0.4.0 and torchvision\
matplotlib\
opencv \
nibabel\
numpy\
tqdm\
scikit-image\
scipy\
pillow\
h5py\

You can install pytorch following instructions in https://pytorch.org/
The other requirements can be installed with pip3 or anaconda

# Citation

If you use any of this code or our ideas, please cite our arxiv paper: https://arxiv.org/abs/1902.04487

# Usage

To run in a given volume, do:
```
python3 run.py path/to/volume.nii.gz
```
It should be able to run if you have all requirements and installed pytorch correctly, in both GPU or CPU.

More instructions in other usages will be available soon.

