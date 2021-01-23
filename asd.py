from hypersiren import *

print_CUDA_memory_statistics()

#put this to true if you want to resume a previous training
restore = False

# SUPER RESOLUTION TEST
datasetTrainingPath = "./images/DIV2K_train_HR"
datasetValidationPath = "./images/DIV2K_valid_HR"
datasetTestPath = "./images/Set14"
epochs = 500 #2000
steps_til_summary = 100
max_num_images = 100 #800
max_val_images = 50 #100
max_test_images = 14
batch_size = 20 #25
hidden_features = 128
better_params = (30, 30, 2e-5, 0.1)
lambda_latent = 1e-1
lambda_weights = 1e2
lambda_biases = 1e2
width_LR = 64
height_LR = 64
factor = 4
width_HR = width_LR * factor
height_HR = height_LR * factor

#checkpoint_path = "/content/drive/MyDrive/backup_training/hyper_init2_SIREN_PRIOR_BRANCH_REG_DA_ep" + str(epochs) + "_im" + str(max_num_images) + ".chk"
#writer_folder = "/content/drive/MyDrive/backup_training/hyper_init2_SIREN_PRIOR_BRANCH_REG_DA_ep" + str(epochs) + "_im" + str(max_num_images)

# set dataloader
training_dataloader = DataLoader(Hyper_ImageFitting_RGB_DA(datasetTrainingPath, width_LR, height_LR, factor, max = max_num_images, apply_random_transforms=True), batch_size=batch_size, pin_memory=True, num_workers=0, shuffle=True)
validation_dataloader = DataLoader(Hyper_ImageFitting_RGB_DA(datasetValidationPath, width_LR, height_LR, factor, max = max_val_images), batch_size=10, pin_memory=True, num_workers=0, shuffle=True)
