# Single Image Super Resolution with SIREN

## High-Level structure
Hypersiren is a python package and is organized as follows:
* dataio.py contains dataloaders for both training and testing data
* training.py contains training routines
* siren_modules.py contains different SIREN modules (with and without hypernetwork)
* hyperalexnet_modules.py contains different hypernetwork modules that uses alexnet as a starting point
* hyper_modules.py contains other hypernetwork modules
* utility.py contains utility functions for tensorboard summaries, checkpoints managing, psnr computing and more
* baselines.py contains functions for computing SISR with state-of-the-art methods

The other folders contains our experiment scripts and the datasets used for training/testing
* experiments_colabs folder contains all our google colabs in which we make our experiments
* images folder contains training, validation and testing datasets for the DIV2K and CelebA datasets
* srgan is a package that contains an implementation of SRGAN and SRResNet

## SRGAN and SRRESNET
The whole code was taken from [HERE](https://github.com/mseitzer/srgan) which is provided under the MIT License                