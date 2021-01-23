from . import *
from .baselines import *
from .dataio import *
from .hyper import *
from .hyperalexnet import *
from .siren import *
from .training import *
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
            sys.exit()
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

'''
function to generate a grid width x height
used as an input for any SIREN
'''
def get_mgrid(width, height, dim=2):
    heightTensor = torch.linspace(-1, 1, steps=height)
    widthTensor = torch.linspace(-1, 1, steps=width)
 
    mgrid = torch.stack(torch.meshgrid(heightTensor, widthTensor), dim=-1)
 
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

'''
function to get a tensor given an image
'''
def get_image_tensor(filename, width_LR=None, height_LR=None, factor=None):
    img = Image.open(filename).convert('RGB')

    if width_LR is None or height_LR is None or factor is None:
        return img
       
    transform_HR = Compose([
        Resize((height_LR * factor, width_LR * factor)),
        ToTensor(),
    ])
 
    transform_LR = Compose([
        Resize((height_LR, width_LR)),
        ToTensor(),
    ])

    # normalize = Compose([
    #     Normalize(torch.Tensor([0.5, 0.5, 0.5]), torch.Tensor([0.5, 0.5, 0.5])) #normalize
    # ])
 
    img_HR = transform_HR(img)
    img_LR = transform_LR(img)
    # img_LR_norm = normalize(img_LR)

    # return img_LR, img_LR_norm, img_HR
    return img_LR, img_HR