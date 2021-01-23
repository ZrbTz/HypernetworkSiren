from hypersiren import *

print_CUDA_memory_statistics()

#put this to true if you want to resume a previous training
restore = False

# SUPER RESOLUTION TEST
datasetTrainingPath = "./images/DIV2K_train_HR"
datasetValidationPath = "./images/DIV2K_valid_HR"
datasetTestPath = "./images/Set14"
epochs = 10 #2000
steps_til_summary = 100
max_num_images = 5 #800
max_val_images = 50 #100
max_test_images = 14
batch_size = 1 #25
hidden_features = 32
better_params = (30, 30, 2e-5, 0.1)
lambda_latent = 1e-1
lambda_weights = 1e2
lambda_biases = 1e2
width_LR = 64
height_LR = 64
factor = 4
width_HR = width_LR * factor
height_HR = height_LR * factor

from srgan import eval

model = "srgan" 
argv = ["-i", "srgan/configs/" + model + ".json", "srgan/resources/pretrained/" + model + ".pth", "./images/DIV2K_train_LR_x8/0001x8.png"]
eval.main(argv)