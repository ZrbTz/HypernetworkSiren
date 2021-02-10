########################################################################
#Convolutional Network using Naive Inception block and residual connection

class ResidualBlock_1(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding = 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding = 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding = 1)  
        )

    def forward(self, x):
        residual = x
        x = self.block(x)
        x += residual
        return x

class InceptionAndResidual(nn.Module):
    def __init__(self, layerSizes):
        super(InceptionAndResidual, self).__init__()
 
        self.layerSizes = layerSizes
 
        #INCEPTION BLOCK
        self.branchConv1x1 = nn.Conv2d(3, 3, kernel_size=1)
        self.branchConv3x3 = nn.Conv2d(3, 3, kernel_size=3, padding = 1)
        self.branchConv5x5 = nn.Conv2d(3, 3, kernel_size=5, padding = 2)
        self.branchAvgPool3x3 = nn.AvgPool2d(3, stride = 1, padding = 1)
        self.afterInc = nn.Sequential(
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=False)
        )

        self.Residual = nn.Sequential( 
            ResidualBlock_1(12),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=False)
        )

        self.lastConv =  nn.Sequential(
            nn.Conv2d(12, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(24, 12, kernel_size=3, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(3, stride=2),
        )

        self.generateWeights = nn.ModuleList()
        
        for layer in self.layerSizes:
            linear = nn.Linear(2700, (layer[0] + 1 ) * layer[1])


            self.generateWeights.append(linear)
 
    def forward(self, image_LR):

        x = image_LR
        #INCEPTION
        x = [
            self.branchConv1x1(x),
            self.branchConv3x3(x),
            self.branchConv5x5(x),
            self.branchAvgPool3x3(x),
        ]
        x = torch.cat(x, 1)
        x = self.afterInc(x)

        #RESIDUAL
        x = self.Residual(x)
        
        #LAST CONV
        x = self.lastConv(x)
        x = torch.flatten(x, 1)

        weights = []
        for generateW in self.generateWeights:
            weights.append(generateW(x))

        weight_outputs = torch.cat(weights, dim=1)
 
        return weight_outputs, None, x
 
    def hyperInceptionAndResidual(layerSizes, pretrained = True, progress = True):
        model = InceptionAndResidual(layerSizes)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                                  progress=progress)
            model.load_state_dict(state_dict, strict=False)
        return model


####################################################################################