from . import *
from .hyperalexnet import HyperMetaAlexNet


#############################################################################################################
# BASIC SIREN #
# This is a basic version of the SIREN, which can have its weights and bias initialized by passing them as paramethers
class Basic_SineLayer(nn.Module):    
    def __init__(self, in_features, out_features, is_first=False, weight = None, bias = None, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=True)
        
        self.init_weights(weight, bias)

    def init_weights(self, weight = None, bias = None):
        with torch.no_grad(): #Context-manager that disables gradient calculation
            if weight == None or bias == None:
                if self.is_first:
                    #Fills the input Tensor with values drawn from the uniform distribution

                    #Initialize weights
                    self.linear.weight.uniform_(-1 / self.in_features, 
                                                1 / self.in_features)      
                else:
                    #Fills the input Tensor with values drawn from the uniform distribution

                    #Initialize weights
                    self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                                np.sqrt(6 / self.in_features) / self.omega_0)
            else:
                self.linear.weight = nn.Parameter(weight)
                self.linear.bias = nn.Parameter(bias)
                
    def forward(self, input): #forward pass
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate

class Basic_Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, w = None, b = None, first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []   #net

        #Initialization without parameters
        if w == None and b == None:
            #append the first sine layer
            self.net.append(Basic_SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))

            #append all the other sine layers
            for i in range(hidden_layers):
                self.net.append(Basic_SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad(): #Context-manager that disables gradient calculation
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        elif b == None:
            w_start = 0
            w_end = w_start + in_features * hidden_features
            b_start = w_end
            b_end = b_start + hidden_features
            self.net.append(Basic_SineLayer(in_features, hidden_features, omega_0=first_omega_0, weight = w[0][w_start : w_end].view(hidden_features,in_features), bias = w[0][b_start : b_end].view(hidden_features)))

            for i in range(hidden_layers):
                w_start = b_end
                w_end = w_start + hidden_features * hidden_features
                b_start = w_end
                b_end = b_start + hidden_features
                self.net.append(Basic_SineLayer(hidden_features, hidden_features, omega_0=hidden_omega_0, weight = w[0][w_start : w_end].view(hidden_features,hidden_features), bias = w[0][b_start : b_end].view(hidden_features)))

            w_start = b_end
            w_end = w_start + hidden_features * out_features
            b_start = w_end
            b_end = b_start + out_features
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight = nn.Parameter(w[0][w_start : w_end].view(out_features,hidden_features))
                final_linear.bias = nn.Parameter(w[0][b_start : b_end].view(out_features))
                
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
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, hypernetInit,
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
 
        self.hypernet = hypernetInit(self.layerSizes, pretrained=False, progress=True)
 
    
    def forward(self, image_LR, coords_HR):  #forward pass
        w, b, latent_space = self.hypernet(image_LR)

        outputs = []
        if b == None:
          for j, weights in enumerate(w):
            w_start = 0
            w_end = w_start + self.layerSizes[0][0] * self.layerSizes[0][1]
            b_start = w_end
            b_end = b_start + self.layerSizes[0][1]
            output = self.net[0](coords_HR[j].view(1,-1,2), weights[w_start : w_end].view(self.layerSizes[0][1],self.layerSizes[0][0]), weights[b_start : b_end].view(self.layerSizes[0][1]))

            for i in range(1, self.hidden_layers+1):
                w_start = b_end
                w_end = w_start + self.layerSizes[i][0] * self.layerSizes[i][1]
                b_start = w_end
                b_end = b_start + self.layerSizes[i][1]
                output = self.net[i](output, weights[w_start : w_end].view(self.layerSizes[i][1],self.layerSizes[i][0]), weights[b_start : b_end].view(self.layerSizes[i][1]))

            w_start = b_end
            w_end = w_start + self.layerSizes[-1][0] * self.layerSizes[-1][1]
            b_start = w_end
            b_end = b_start + self.layerSizes[-1][1]
            outputs.append(self.net[-1](output, weights[w_start : w_end].view(self.layerSizes[-1][1],self.layerSizes[-1][0]), weights[b_start : b_end].view(self.layerSizes[-1][1])))

        else:
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
