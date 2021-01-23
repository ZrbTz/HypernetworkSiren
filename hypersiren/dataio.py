from . import *
from .baselines import *
from .hyper import *
from .hyperalexnet import *
from .siren import *
from .training import *
from .utility import *

########################################################################################
# TRAINING DATALOADER WITH DATA AUGMENTATION #

class Hyper_ImageFitting_RGB_DA(Dataset):
    def __init__(self, path, width_LR, height_LR, factor, max=200, apply_random_transforms = False):
        super().__init__()
        self.width_LR = width_LR
        self.height_LR = height_LR
        self.factor = factor
        self.applyRandomTransform = apply_random_transforms

        self.dataset = {}
        self.counter = 0
        images = os.listdir(path)

        self.pre_transform = Compose([
            RandomResizedCrop((height_LR * factor, width_LR * factor), scale=(0.25, 1.0)),

            RandomApply(torch.nn.ModuleList([
                Pad(padding = 50, padding_mode = "reflect"),
                RandomRotation(30, resample = 0),
                CenterCrop((height_LR * factor, width_LR * factor)),
            ]), p=0.3),

            RandomApply(torch.nn.ModuleList([
                ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            ]), p=0.3),
            
            RandomHorizontalFlip(p=0.3),
            RandomVerticalFlip(p=0.3),
            RandomGrayscale(p=0.1)       
        ])
          
        self.transform_HR = Compose([
            Resize((height_LR * factor, width_LR * factor)),
            ToTensor(),
            #Normalize(torch.Tensor([0.5, 0.5, 0.5]), torch.Tensor([0.5, 0.5, 0.5])) #normalize
        ])
    
        self.transform_LR = Compose([
            Resize((height_LR, width_LR)),
            ToTensor(),
            #Normalize(torch.Tensor([0.5, 0.5, 0.5]), torch.Tensor([0.5, 0.5, 0.5])) #normalize
        ])

        for img in images:
          imagePath = path + "/" + img
          self.dataset[self.counter] = get_image_tensor(imagePath)
          self.counter += 1
          if self.counter == max:
            break

    def __len__(self):
        return self.counter

    def __getitem__(self, idx):    
        img = self.dataset[idx]
    
        if self.applyRandomTransform:
            img = self.pre_transform(img)

        img_HR = self.transform_HR(img)
        img_LR = self.transform_LR(img)

        pixels_LR = img_LR
        pixels_HR = img_HR.permute(1, 2, 0).view(-1, 3)
        coords_HR = get_mgrid(self.height_LR*self.factor, self.width_LR*self.factor)
            
        return coords_HR, pixels_HR, pixels_LR

#####################################################################################
#TRAINING DATA LOADER WITHOUT DATA AUGMENTATION#

class Hyper_ImageFitting_RGB(Dataset):
    def __init__(self, path, width_LR, height_LR, factor, max=200):
        super().__init__()
        self.width_LR = width_LR
        self.height_LR = height_LR
        self.factor = factor

        self.dataset = {}
        self.counter = 0
        images = os.listdir(path)

        for img in images:
          imagePath = path + "/" + img
          self.dataset[self.counter] = get_image_tensor(imagePath, width_LR, height_LR, factor)
          self.counter += 1
          if self.counter == max:
            break

    def __len__(self):
        return self.counter

    def __getitem__(self, idx):    
        img_LR, img_HR = self.dataset[idx]

        pixels_LR = img_LR
        pixels_HR = img_HR.permute(1, 2, 0).view(-1, 3)
        coords_HR = get_mgrid(self.height_LR*self.factor, self.width_LR*self.factor)
            
        return coords_HR, pixels_HR, pixels_LR


##############################################################################################
#TEST DATALOADER#
class TestImageFitting_RGB(Dataset):
    def __init__(self, path, width_LR, height_LR, factor, max=200):
        super().__init__()
        self.width_LR = width_LR
        self.height_LR = height_LR
        self.factor = factor

        self.dataset = {}
        self.counter = 0
        images = os.listdir(path)

        for img in images:
          imagePath = path + "/" + img
          self.dataset[self.counter] = (get_image_tensor(imagePath, width_LR, height_LR, factor), imagePath)
          self.counter += 1
          if self.counter == max:
            break

    def __len__(self):
        return self.counter

    def __getitem__(self, idx):   
        image, filename = self.dataset[idx]

        img_LR, img_HR = image

        pixels_LR_original = img_LR
        pixels_LR = img_LR.permute(1, 2, 0).view(-1, 3)
        pixels_HR = img_HR.permute(1, 2, 0).view(-1, 3)

        coords_LR = get_mgrid(self.height_LR, self.width_LR)
        coords_HR = get_mgrid(self.height_LR*self.factor, self.width_LR*self.factor)
            
        return coords_LR, coords_HR, pixels_HR, pixels_LR, pixels_LR_original, filename