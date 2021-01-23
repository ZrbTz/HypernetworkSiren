from . import *
from .utility import *

'''
Function to get an upscaled image using the bicubip interpolation method
'''
def get_bicubic_image(filename, lrwdim, lrhdim, hrwdim, hrhdim):
  img = Image.open(filename)

  transform = Compose([
      Resize((lrhdim,lrwdim), interpolation=Image.BICUBIC),
      Resize((hrhdim,hrwdim), interpolation=Image.BICUBIC),
      ToTensor(),
  ])
  img = transform(img)
  return img

'''
Script to get an upscaled image using the SRGAN method
NOTE: The input image format must be .png"
'''
def getSRGAN(imageName):
    model = "srgan" 
    argv = ["-i", "configs/" + model + ".json", "resources/pretrained/" + model + ".pth", imageName]
    eval.main(argv)
    imageName = imageName[:-4]
    imageName = imageName + "_srgan.png"
    srgan = get_image_tensor(imageName, height_HR, width_HR,1)[0].permute(1, 2, 0).view(-1, 3)
    return srgan

'''
Script to get an upscaled image using the SRResNet method
NOTE: The input image format must be .png"
'''
def getSRRESNET(imageName):
    model = "srresnet" 
    argv = ["-i", "configs/" + model + ".json", "resources/pretrained/" + model + ".pth", imageName]
    eval.main(argv)
    imageName = imageName[:-4]
    imageName = imageName + "_resnet.png"
    srresnet = get_image_tensor(imageName, height_HR, width_HR,1)[0].permute(1, 2, 0).view(-1, 3)
    return srresnet