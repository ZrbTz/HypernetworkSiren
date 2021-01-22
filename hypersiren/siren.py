import torch
from torch import nn

#############################################################################################################
# BASIC SIREN #
# This is a basic version of the SIREN, which can have its weights and bias initialized by passing them as paramethers

class Basic_Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, w = None, b = None, first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []   #net

        #Initialization without parameters
        if w == None:
            #append the first sine layer
            self.net.append(Basic_SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))

            #append all the other sine layers
            for i in range(hidden_layers):
                self.net.append(Basic_SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad(): #Context-manager that disables gradient calculation
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        
        #Initialization with paramethers
        else:
            w_start = 0
            w_end = w_start + in_features * hidden_features
            b_start = 0
            b_end = b_start + hidden_features
            self.net.append(Basic_SineLayer(in_features, hidden_features, omega_0=first_omega_0, weight = w[0][w_start : w_end].view(hidden_features,in_features), bias = b[0][b_start : b_end].view(hidden_features)))

            for i in range(hidden_layers):
                w_start = w_end
                w_end = w_start + hidden_features * hidden_features
                b_start = b_end
                b_end = b_start + hidden_features
                self.net.append(Basic_SineLayer(hidden_features, hidden_features, omega_0=hidden_omega_0, weight = w[0][w_start : w_end].view(hidden_features,hidden_features), bias = b[0][b_start : b_end].view(hidden_features)))

            w_start = w_end
            w_end = w_start + hidden_features * out_features
            b_start = b_end
            b_end = b_start + out_features
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight = nn.Parameter(w[0][w_start : w_end].view(out_features,hidden_features))
                final_linear.bias = nn.Parameter(b[0][b_start : b_end].view(out_features))
                
            self.net.append(final_linear)            

        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords): 
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords) #apply coords on net
        return output, coords

#############################################################################################################
# HYPER SIREN #
# This version of the SIREN generates weights and bias using an hypernetwork
#Todo: decouple this and alexnet hypernetwork

class Hyp_SineLayer(nn.Module):    
    def __init__(self, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        
    def forward(self, input, weight, bias): #forward pass
        return torch.sin(self.omega_0 * torch.nn.functional.linear(input, weight, bias))
 
 
class Hyp_LinearLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input, weight, bias): #forward pass
        return torch.nn.functional.linear(input, weight, bias)
      
    
class Hyp_Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
 
        self.hidden_layers = hidden_layers
 
        self.layerSizes = []
        self.net = nn.ModuleList()
 
        self.layerSizes.append([in_features,hidden_features])
        self.net.append(Hyp_SineLayer(omega_0=first_omega_0))
 
        for i in range(self.hidden_layers):
          self.layerSizes.append([hidden_features,hidden_features])
          self.net.append(Hyp_SineLayer(omega_0=hidden_omega_0))
 
        self.layerSizes.append([hidden_features,out_features])
        self.net.append(Hyp_LinearLayer())

        self.hypernet = HyperAlexNet.hyperAlexNet(self.layerSizes, pretrained=False, progress=True)
    
    def forward(self, image_LR, coords_HR):
        #Let the hypernetwork generate our weights and bias for each layer
        w, b, latent_space = self.hypernet(image_LR)

        #Use the SIREN layers to generate the output, given w and b from the hypernet
        outputs = []
        for j, weights in enumerate(w):
          w_start = 0
          w_end = w_start + self.layerSizes[0][0] * self.layerSizes[0][1]
          b_start = 0
          b_end = b_start + self.layerSizes[0][1]
          output = self.net[0](coords_HR[j].view(1,-1,2), weights[w_start : w_end].view(self.layerSizes[0][1],self.layerSizes[0][0]), b[j][b_start : b_end].view(self.layerSizes[0][1]))

          for i in range(1, self.hidden_layers+1):
              w_start = w_end
              w_end = w_start + self.layerSizes[i][0] * self.layerSizes[i][1]
              b_start = b_end
              b_end = b_start + self.layerSizes[i][1]
              output = self.net[i](output, weights[w_start : w_end].view(self.layerSizes[i][1],self.layerSizes[i][0]), b[j][b_start : b_end].view(self.layerSizes[i][1]))

          w_start = w_end
          w_end = w_start + self.layerSizes[-1][0] * self.layerSizes[-1][1]
          b_start = b_end
          b_end = b_start + self.layerSizes[-1][1]
          outputs.append(self.net[-1](output, weights[w_start : w_end].view(self.layerSizes[-1][1],self.layerSizes[-1][0]), b[j][b_start : b_end].view(self.layerSizes[-1][1])))

        outputs = torch.cat(outputs)

        return outputs, coords_HR, w, b, latent_space
