from . import *
########################################################################################################################
# HYPERNETWORK TRAINING #
# Training of hypersyren, the hypernetwork will be trained to give our SIREN a good prior


def prior_train(net, dataloader, validationDataLoader, writer, lr=1e-4, gamma=0.1, step = 0, total_steps = 500,
                steps_til_summary = 5, width_HR = 256, height_HR = 256, lambda_latent = 0, lambda_weights = 0,
                restore=False, checkpoint_path=""):
  #define optimizer
  params_to_optimize = net.parameters()
  optimizer = torch.optim.Adam(lr=lr, params=params_to_optimize, weight_decay=0.0005)
  scheduler= torch.optim.lr_scheduler.StepLR(optimizer, total_steps//2, gamma=gamma)

  step = restoreModel(net = net, scheduler = scheduler, optimizer = optimizer, restore = restore, checkpoint_path = checkpoint_path)
 
  estimator = Time_estimator(total_steps, steps_til_summary)
  loss = 0

  while step < total_steps:
      psnrs_t = []
      for input_grid_HR, ground_truth_HR, ground_truth_LR in dataloader:
 
          input_grid_HR, ground_truth_HR, ground_truth_LR = input_grid_HR.cuda(), ground_truth_HR.cuda(), ground_truth_LR.cuda()
 
          output_image_HR, coords_HR, weights, bias, latent_space = net(ground_truth_LR, input_grid_HR) #forward pass 
 
          loss = ((output_image_HR - ground_truth_HR)**2).mean()   #calculate loss on image (MSE)
          loss += lambda_latent * ((latent_space)**2).mean()  #loss on latent space (enforces a Gaussian prior on latent code)
          loss += lambda_weights * ((weights)**2).mean()   #loss on weights (encourage a lower frequency representation of the image)

          psnrs_t.append(psnr(input = output_image_HR, target = ground_truth_HR))

          input_grid_HR.cpu(), ground_truth_HR.cpu(), ground_truth_LR.cpu(), output_image_HR.cpu(), coords_HR.cpu()
          del input_grid_HR
          del ground_truth_HR
          del ground_truth_LR
          del output_image_HR
          del coords_HR

          optimizer.zero_grad() #Sets the gradients of all optimized torch.Tensor to zero
          loss.backward() #apply gradients
          optimizer.step()  #make optimizer step
      
      writer.add_scalar("loss_prior_train", loss.item(), step)

      if step % steps_til_summary == steps_til_summary-1:
          print_CUDA_memory_statistics()

          psnrs_v = []
          for input_grid_HR, ground_truth_HR, ground_truth_LR in validationDataLoader:

              input_grid_HR, ground_truth_HR, ground_truth_LR = input_grid_HR.cuda(), ground_truth_HR.cuda(), ground_truth_LR.cuda()

              output_image_HR, coords_HR, _, _, _ = net(ground_truth_LR, input_grid_HR)              

              psnrs_v.append(psnr(input = output_image_HR, target = ground_truth_HR))

              input_grid_HR.cpu(), ground_truth_HR.cpu(), ground_truth_LR.cpu(), output_image_HR.cpu(), coords_HR.cpu()
              del input_grid_HR
              del ground_truth_HR
              del ground_truth_LR
              del output_image_HR
              del coords_HR

          psnr_train = sum(psnrs_t) / len(psnrs_t)
          psnr_val = sum(psnrs_v) / len(psnrs_v)

          writer.add_scalars("psnr_prior",{
              "train": psnr_train,
              "val": psnr_val
          }, step)
          writer.flush();
          
          print("Step %d, Total loss %0.6f, psnr_train: %0.6f, psnr_val: %0.6f" % (step+1, loss, psnr_train, psnr_val))

          estimator.checkpoint(step)

          saveModel(net = net, optimizer = optimizer, scheduler = scheduler, step = step, checkpoint_path = checkpoint_path)

      scheduler.step()
      step = step + 1


###################################################################################################################################################
# TRAINING OF A BASIC SIREN #
# This function trains a basic SIREN 

def train(net, model_input, ground_truth, writer, name, lr=1e-4, gamma=0.1, total_steps = 1500, steps_til_summary = 100, width = 64, height = 64, useL1 = False, useL2 = False):
  #define optimizer
  optimizer = torch.optim.Adam(lr=lr, params=net.parameters(), weight_decay=0.0005)
  scheduler= torch.optim.lr_scheduler.StepLR(optimizer, 800, gamma=gamma)
  
  # training
  for step in range(total_steps):

      #model_output is the output of the single forward pass, it is an output image
      #tensor of dimension (sidelength x sidelength, 1);
      #this represents the y^ (output) value for each pixel

      #coords is a copy of the model_input pairs
      model_output, coords = net(model_input) #forward pass 

      #model_output and ground_truth (original image) have the same dimensions
      loss = ((model_output - ground_truth)**2).mean()  #calculate loss (MSE)
      
      #L1 regularization
      if(useL1):
        reg_lambda = 0.01
        L1_reg = torch.tensor(0., requires_grad=True)
        for name, param in net.named_parameters():
            if 'weight' in name:
                L1_reg = L1_reg + torch.norm(param, 1)
        loss = loss + reg_lambda * L1_reg

      #L2 regularization
      if(useL2):
        reg_lambda = 0.0000201
        l2_reg = 0
        num = 0
        for W in net.parameters():
            l2_reg += W.norm(2)**2
            #num += W.
        loss += l2_reg*reg_lambda

      writer.add_scalar("loss_fit_" + name, loss.item(), step)

      if step % steps_til_summary == steps_til_summary-1:
          writer.add_image("img_fit_" + name, torch.clamp(model_output.cpu(), min=0, max=1).view(height,width,3).detach().numpy(), step, dataformats='HWC')

      optimizer.zero_grad() #Sets the gradients of all optimized torch.Tensor to zero
      loss.backward() ##apply gradients
      optimizer.step()  #make optimizer step
      scheduler.step()