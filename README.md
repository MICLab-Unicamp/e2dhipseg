# Extended 2D Consensus Hippocampus Segmentation (e2dhipseg) - beta

# Introduction
This work implements the segmentation method proposed in the paper Hippocampus segmentation on epilepsy and Alzheimer's disease studies with multiple convolutional neural networks published in Heliyon, Volume 7, Issue 2, 2021
(https://www.sciencedirect.com/science/article/pii/S2405844021003315)

A masters dissertation related to this work has been published:
Deep Learning for Hippocampus Segmentation (http://repositorio.unicamp.br/handle/REPOSIP/345970)

This repository is also the official implementation for the short paper Extended 2D Consensus Hippocampus Segmentation presented at the International Conference on Medical Imaging with Deep Learning (MIDL), 2019

Authors: Diedre Carmo, Bruna Silva, Clarissa Yasuda, Leticia Rittner, Roberto Lotufo

This code is mainly intended to be used for others to be able to run this in a given MRI dataset or file. Notice we published a standalone release with a minimal GUI for ease of prediction. Any problem please create an issue and i will try to help you.

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
For CPU only installation on Windows use the following commands:
- pip install torch==1.3.0+cpu -f https:&#8203;//download.pytorch.org/whl/torch_stable.html
- pip install torchvision==0.2.2.post3

A Windows standalone executable can be compiled with:
- pyinstaller e2dhipse_windows.spec

FLIRT can be installed following https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation/Linux (for Windows executables are aleady included)
The other requirements can be installed with pip3


# Citation

If you use any of this code or our ideas, please cite one of these publications:

BibTex\
Preprint:

Journal Publication:

    @article{CARMO2021e06226,
    title = {Hippocampus segmentation on epilepsy and Alzheimer's disease studies with multiple convolutional neural networks},
    journal = {Heliyon},
    volume = {7},
    number = {2},
    pages = {e06226},
    year = {2021},
    issn = {2405-8440},
    doi = {https://doi.org/10.1016/j.heliyon.2021.e06226},
    url = {https://www.sciencedirect.com/science/article/pii/S2405844021003315},
    author = {Diedre Carmo and Bruna Silva and Clarissa Yasuda and Letícia Rittner and Roberto Lotufo},
    keywords = {Deep learning, Hippocampus segmentation, Convolutional neural networks, Alzheimer's disease, Epilepsy},
    abstract = {Background: Hippocampus segmentation on magnetic resonance imaging is of key importance for the diagnosis, treatment decision and investigation of neuropsychiatric disorders. Automatic segmentation is an active research field, with many recent models using deep learning. Most current state-of-the art hippocampus segmentation methods train their methods on healthy or Alzheimer's disease patients from public datasets. This raises the question whether these methods are capable of recognizing the hippocampus on a different domain, that of epilepsy patients with hippocampus resection. New Method: In this paper we present a state-of-the-art, open source, ready-to-use, deep learning based hippocampus segmentation method. It uses an extended 2D multi-orientation approach, with automatic pre-processing and orientation alignment. The methodology was developed and validated using HarP, a public Alzheimer's disease hippocampus segmentation dataset. Results and Comparisons: We test this methodology alongside other recent deep learning methods, in two domains: The HarP test set and an in-house epilepsy dataset, containing hippocampus resections, named HCUnicamp. We show that our method, while trained only in HarP, surpasses others from the literature in both the HarP test set and HCUnicamp in Dice. Additionally, Results from training and testing in HCUnicamp volumes are also reported separately, alongside comparisons between training and testing in epilepsy and Alzheimer's data and vice versa. Conclusion: Although current state-of-the-art methods, including our own, achieve upwards of 0.9 Dice in HarP, all tested methods, including our own, produced false positives in HCUnicamp resection regions, showing that there is still room for improvement for hippocampus segmentation methods when resection is involved.}
    }

# Binary Release
To make this easier to use, a standalone binary compilation of the code is available as a [release](https://github.com/MICLab-Unicamp/e2dhipseg/releases).

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

