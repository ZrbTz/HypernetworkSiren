# Single Image Super Resolution with SIREN

This work was made by: Simone Cavallera, Michele Crepaldi, Thomas Madeo
our report can be found here https://it.overleaf.com/1191221391dchmtqmqmjkq
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

## Experiments

There are 5 esperiments in ipynb files that we tested on google colab
* ImageFitting_Poisson: In this experiment we try the basic siren capabilities by fitting an image and trying the poisson reconstruction both from the gradient and the laplacian
* Ablation_Study: In this experiment we changed the siren structure to check how it reacts
* SirenSISR: In this experiment we try the vanilla SIREN on the Single Image Super Resolution Task and we compare it to other networks
* Hypernetwork: In this experiment we try an hypernetwork to give a prior to our siren, there are multiple hypernetworks that can be tried by changing the hypernetInit paramether
* Fitting_Hypernetwork: In this experiment we try to fit the hypernetwork directly on the single image

You can try different hypernetworks by changing the hypernetInit paramether with one of the following:
* VGG: vgg19
* AlexNet: HyperBaseAlexNet.hyperBaseAlexNet
* AlexNet with one FC for each SIREN layers: HyperBaseAlexNetFC.hyperBaseAlexNetFC
* AlexNet with one FC for each weight and bias of SIREN layers: HyperMetaAlexNet.hyperMetaAlexNet
* Hypernetwork with inception block and residual connection: InceptionAndResidual.hyperInceptionAndResidual
* Alexnet with a residual conjAlexWithResidual.hyperAlexWithResidual

## SRGAN and SRRESNET
The whole code was taken from [HERE](https://github.com/mseitzer/srgan) which is provided under the MIT License                
