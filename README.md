# Extended 2D Consensus Hippocampus Segmentation (e2dhipseg) - beta

# Introduction
The masters dissertation related to this work has been published:
Deep Learning for Hippocampus Segmentation (http://repositorio.unicamp.br/handle/REPOSIP/345970)

This repository is also the official implementation for the papers:

Journal pre-print:\
Hippocampus Segmentation on Epilepsy and Alzheimer’s Disease Studies with Multiple Convolutional Neural Networks (https://arxiv.org/pdf/2001.05058.pdf)

Short Paper:\
Extended 2D Consensus Hippocampus Segmentation (https://arxiv.org/abs/1902.04487) published at the International Conference on edical Imaging with Deep Learning, 2019

Authors: Diedre Carmo, Bruna Silva, Clarissa Yasuda, Leticia Rittner, Roberto Lotufo

Please note the code may be confusing in some places due to containing backwards compatibility to many experiments, and being my
personal Deep Learning in medical imaging code repository.

Improvements are still being made privately, and will be made public when the corresponding paper is published.
Please cite one of our papers if you use code from this repository!

This is a beta release, any problem please create an issue and i will try to help you!

Thank you!

# Minimum Requirements
At least 8GB of RAM
Ubuntu 16.04 or 18.04, might work in other distributions but not tested

Having a GPU is not necessary, but will speed prediction time per volume dramatically (from around 5 minutes in CPU to 15 seconds in a 1060 GPU).


# Software Requirements
If you are using the binary release, no enviroment setup should be necessary besides FLIRT, if used.

To run this code, you need to have the following libraries installed:

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
FLIRT - optional if you want to use the automated volume registration, please do use it if your volumes are not in the MNI152
orientation, or you will get blank/wrong outputs\

You can install pytorch following the guide in in https://pytorch.org/. If you plan to use a GPU, you should have the correct CUDA and CuDNN for your pytorch installation.

FLIRT can be installed following https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation/Linux
The other requirements can be installed with pip3


# Citation

If you use any of this code or our ideas, please cite one of these publications:

BibTex\
Preprint:

Journal Preprint:\
@misc{carmo2020hippocampus,
    title={Hippocampus Segmentation on Epilepsy and Alzheimer's Disease Studies with Multiple Convolutional Neural Networks},
    author={Diedre Carmo and Bruna Silva and Clarissa Yasuda and Letícia Rittner and Roberto Lotufo},
    year={2020},
    eprint={2001.05058},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}\

Short Paper:\
@inproceedings{carmo2019midl,
  title={Extended 2D Volumetric Consensus Hippocampus Segmentation},
  author={Carmo, Diedre and Silva, Bruna and Yasuda, Clarissa and Rittner, Let{\'\i}cia and Lotufo, Roberto},
  booktitle={International Conference on Medical Imaging with Deep Learning},
  year={2019}
}\

# Binary Release (alpha)
To make this easier to use, a standalone binary compilation of the code is available at:

Binary release v0.1.a: https://drive.google.com/file/d/112nKUpn0sQurn1Whj2FDzFDg8HnhKJgX/view?usp=sharing

Download, unpack the .zip file and run the "run" file. You don't need to use "sudo". No setup of enviroment should be needed,
only the installation of FLIRT if you use the registration option. This is in an alpha release, if you find any problem,
please create an issue.

# Usage
If you are using the binary release version, replace ```python3 run.py``` with ```./run.py```.\
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
Expected June 2020
-Incoming final version of the method, delayed by the current world situation.

-January 2020
-Added compiled executable link, updated to journal pre-print version

October 10 2019
-Updated weights for better performance

April 4 2019
-Added quality of life options, FLIRT registration and back to original space, run on folder, improvements and bug fixes.

Feb 8 2019
-Initial commit
