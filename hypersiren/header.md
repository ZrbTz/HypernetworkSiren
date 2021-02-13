# HEADER
A description of all modules and submodules contained in this package

## UTILITY
Contains some useful function

| Function | Description |
| ------ | ----------- |
| **psnr(input, target)**  | Computes the psnr value between two tensors |
| **saveModel(net, optimizer, scheduler, step, checkpoint_path)** | Saves the state of the network |
| **restoreModel(net, scheduler, optimizer, restore=False, checkpoint_path="")** | Restores the state of the network |
| **print_CUDA_memory_statistics()** | prints GPU memory usage |
| **get_mgrid(width, height, dim=2)** | generates a grid |
| **get_image_tensor(filename, width_LR=None, height_LR=None, factor=None)**|   gets tensor from an image|

### Classes
+ **TimeEstimator()**: Utility class to estimate training time 

## DATAIO
Contais dataloaders

| Class | Description |
| ------ | ----------- |
| **Hyper_ImageFitting_RGB_DA(self, path, width_LR, height_LR, factor, max=200, apply_random_transforms = False)**  | DataLoader used for training which includes data augmentation |
| **Hyper_ImageFitting_RGB(self, path, width_LR, height_LR, factor, max=200)** | DataLoader used for training|
| **TestImageFitting_RGB(self, path, width_LR, height_LR, factor, max=200)** | DataLoader used for testing |

## HYPERALEXNET
Contains hypernetwork models based on alexnet

| Class | Description |
| ------ | ----------- |
| **HyperMetaAlexNet(self, layerSizes)**  | The network that gave us the best results |
| **HyperBaseAlexNet(self, layerSizes)** | Base AlexNet|

## BASELINES
Contains functions used to compare our results with other methods

| Function | Description |
| ------ | ----------- |
| **get_bicubic_image(filename, lrwdim, lrhdim, hrwdim, hrhdim)**  | Applies bicupic interpolation to an image |
| **getSRGAN(imageName)** | Gets super resolution using a pretrained SRGAN|
| **getSRRESNET(imageName)** | Gets super resolution using a pretrained SRResNet|

## SIREN
Contains the networks based on SIREN

| Class | Description |
| ------ | ----------- |
| **Basic_Siren(self, in_features, hidden_features, hidden_layers, out_features, w = None, b = None, first_omega_0=30, hidden_omega_0=30.)**  | Basic Siren |
| **Hyp_Siren(self, in_features, hidden_features, hidden_layers, out_features, first_omega_0=30, hidden_omega_0=30.)** | SIREn that uses an hypernetwork|

## TRAINING
Contains training routines

| Function | Description |
| ------ | ----------- |
| **prior_train(net, dataloader, validationDataLoader, writer, lr=1e-4, gamma=0.1, step = 0, total_steps = 500 ,steps_til_summary = 5, width_HR = 256, height_HR = 256, lambda_latent = 0, lambda_weights = 0, restore=False, checkpoint_path="")**  | Trains an hypernetwork |
| **train(net, model_input, ground_truth, writer, name, lr=1e-4, gamma=0.1, total_steps = 1500, steps_til_summary = 100, width = 64, height = 64, useL1 = False, useL2 = False)** | Trains the basic siren|


