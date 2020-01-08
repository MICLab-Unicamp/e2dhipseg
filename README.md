# e2dhipseg

# Introduction
This contains official implementation for Extended 2D Consensus Hippocampus Segmentation 
Pre-print: (https://arxiv.org/abs/1902.04487) and an
Extended Abstract published at the International Conference on edical Imaging with Deep Learning

We made this version of the code public for the following publication: 
Authors: Diedre Carmo, Bruna Silva, Clarissa Yasuda, Leticia Rittner, Roberto Lotufo
Title: Extended 2D Consensus Hippocampus Segmentation
In: International Conference on edical Imaging with Deep Learning, 2019

We are implementing many quality of life changes actively, updates for this repository will be regular.
Improvements are also still being made privately.

Thank you!

# Minimum Requirements
At least 8GB of RAM
Ubuntu 16.04 or 18.04, might work in other distributions but not tested

Having a GPU is not necessary, but will speed prediction time per volume dramatically (from around 5 minutes in CPU to 15 seconds in a 1060 GPU).


# Software Requirements
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
FLIRT (optional if you want to pre-register volumes)\

You can install pytorch following the guide in in https://pytorch.org/. If you plan to use a GPU, you should have the correct CUDA and CuDNN for your pytorch installation.

FLIRT can be installed following https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation/Linux
The other requirements can be installed with pip3 or anaconda

# Citation

If you use any of this code or our ideas, please cite our arxiv paper: https://arxiv.org/abs/1902.04487 or our Extended Abstract: 

BibTex
Preprint:
@article{carmo2019extended,
  title={Extended 2D Volumetric Consensus Hippocampus Segmentation},
  author={Carmo, Diedre and Silva, Bruna and Yasuda, Clarissa and Rittner, Let{\'\i}cia and Lotufo, Roberto},
  journal={arXiv preprint arXiv:1902.04487},
  year={2019}
}

Extended Abstract:
@inproceedings{carmo2019midl,
  title={Extended 2D Volumetric Consensus Hippocampus Segmentation},
  author={Carmo, Diedre and Silva, Bruna and Yasuda, Clarissa and Rittner, Let{\'\i}cia and Lotufo, Roberto},
  booktitle={International Conference on Medical Imaging with Deep Learning},
  year={2019}
}

# Usage
To run in a single volume using a simple graphical interface, do:
```
python3 run.py
```

To run in a given volume, do:
```
python3 run.py path/to/volume.nii.gz
```

To run in all volumes in a given directory use -dir, e.g.:
```
python3 run.py path/to/folder -dir
```

To perform MNI152 registration before segmentation (the resulting mask will be back in the original space), use -reg, e.g.:
```
python3 run.py path/to/folder -dir -reg
```
Registration might improve results, at the cost of 1 minute per volume in runtime.

To display the results of the segmentation using OpenCV, use -display, e.g.:
```
python3 run.py path/to/volume.nii.gz -reg -display
```
Press S to stop in a slice, R to revert direction, press up or down (numpad or directional) to reduce and improve "travelling" speed when displaying slices.

ARGUMENTS SHOULD COME AFTER THE INPUT PATH, as in the examples. All three arguments work in any combination. It should be able to run if you have all requirements and installed pytorch correctly, in both GPU or CPU.

We highly recommend that you use the -reg option, making sure your FLIRT installation is working. If you dont use -reg, make sure your volume is in the MNI152 head orientation, or you might get wrong/empty results. Any problems you can create a issue here, and i will have a look!

# Updates
Expected March 2020
-Incoming final version of the method

October 10 2019
-Updated weights for better performance

April 4 2019
-Added quality of life options, FLIRT registration and back to original space, run on folder, improvements and bug fixes.

Feb 8 2019
-Initial commit
