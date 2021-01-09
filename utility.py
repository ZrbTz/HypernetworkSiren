import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import datetime, os
 
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, ToPILImage
import numpy as np
import skimage
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from math import log10, sqrt
from torch.nn.functional import mse_loss

import time

"""
Function to be used to calculate the psnr between an input image and a target image
"""
def psnr(input, target):
    MSE = ((input - target)**2).mean() 
    PSNR_reconstructed_image = 20*log10(1/sqrt(MSE.item()))
    return PSNR_reconstructed_image
    
"""
Time estimator class; to be initialized outside the training loop with the total steps and the steps_til_summary.
Inside the training loop then call multiple times the checpoint method (passing the current step); this will print
the actual passed/total time estimated
"""
class Time_estimator():
    def __init__(self, total_steps, steps_til_summary):

        self.start = time.time()

        self.total_steps = total_steps
        self.steps_til_summary = steps_til_summary
        
    def checkpoint(self, step):
        self.end = time.time()

        diff = self.end - self.start
        time_passed = diff * (step / self.steps_til_summary)
        estimated_total_time = diff * (self.total_steps / self.steps_til_summary)

        hour_passed = int(time_passed // 3600)
        min_passed = int((time_passed - hour_passed * 3600) // 60)
        sec_passed = int(time_passed - hour_passed * 3600 - min_passed * 60)

        est_tot_hour = int(estimated_total_time // 3600)
        est_tot_min = int((estimated_total_time - est_tot_hour * 3600) // 60)
        est_tot_sec = int(estimated_total_time - est_tot_hour * 3600 - est_tot_min * 60)

        print("Estimated time: {}:{}:{} / {}:{}:{}".format(hour_passed, min_passed, sec_passed, est_tot_hour, est_tot_min, est_tot_sec))

        self.start = time.time()
        
"""
Function to save a network state to file.
It needs as input the net, optimizer, scheduler used in the training procedure together with the
last completed step (not step+1) and the name of the file (checkpoint_path) where to save the checkpoint
"""
def saveModel(net, optimizer, scheduler, step, checkpoint_path):
    print('=> Saving model... ', end='')
    state = {
        'net': net.state_dict(),
        'step': step + 1,
        'optimizer' : optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(state, checkpoint_path)
    print("Saved")

"""
Function to restore the state of the model (and of the scheduler and optimizer) from file.
It requires the net, optimizer, scheduler used in the training procedure together with a boolean
(restore) indicating if to restore from file or not and the name of the file (checkpoint_path)
from which retrieve the data.
It will output the initial step from which to continue/start training.
"""
def restoreModel(net, scheduler, optimizer, restore=False, checkpoint_path=""):
    if restore:
        if os.path.isfile(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            step = checkpoint['step']
            scheduler.load_state_dict(checkpoint['scheduler'])
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})" .format(checkpoint_path, checkpoint['step']))

            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_path))
            exit()
    else:
        step = 0

    return step
     
"""
Function to print the CUDA memory statistics of Colab
"""
def print_CUDA_memory_statistics():
    allocated = torch.cuda.memory_allocated()
    total = torch.cuda.get_device_properties(0).total_memory
    print("CUDA-0 used memory [{:} bytes / {:} bytes ({:.2%})]".format(allocated, total,allocated/total))
