from . import *

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
 
    def hyperInceptionAndResidual(layerSizes, pretrained = False, progress = True):
        model = InceptionAndResidual(layerSizes)
        return model


########################################################################
# VGG Convolutional Network
model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

class VGG(nn.Module):

    def __init__(self, features, layerSizes, init_weights=True):
        super(VGG, self).__init__()

        self.layerSizes = layerSizes
        '''self.totalNumberOfWeights = sum(x[0] * x[1] for x in self.layerSizes)
        self.totalNumberOfBiases = sum(x[1] for x in self.layerSizes)'''

        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.generateWeights = nn.ModuleList()

        for layer in self.layerSizes:
            self.generateWeights.append(nn.Linear(512 * 7 * 7, (layer[0] + 1 ) * layer[1]))
            
        '''self.generateWeights = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, self.totalNumberOfWeights + self.totalNumberOfBiases),
        )'''
    
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        weights = []
        for generateW in self.generateWeights:
            weights.append(generateW(x))

        weight_outputs = torch.cat(weights, dim=1)

        '''weight_outputs = self.generateWeights(x)'''
 
        return weight_outputs, None, None

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg,  layerSizes, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm),  layerSizes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def vgg19(layerSizes, pretrained=False, progress=True, **kwargs):
    return _vgg('vgg19', 'E',  layerSizes, False, pretrained, progress, **kwargs)

##########################################################################################

class ResidualBlock_2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 1):
        super().__init__()

        self.block = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    
    def forward(self, x):
        residual = x
        x = self.block(x)
        x += residual
        return x
 
class AlexWithResidual(nn.Module):
    def __init__(self, layerSizes):
        super(AlexWithResidual, self).__init__()
 
        self.layerSizes = layerSizes
        self.totalNumberOfWeights = sum(x[0] * x[1] for x in self.layerSizes)
        self.totalNumberOfBiases = sum(x[1] for x in self.layerSizes)
 
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ResidualBlock_2(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.generateWeights = nn.ModuleList()

        for layer in self.layerSizes:
            self.generateWeights.append(nn.Linear(256 * 6 * 6, (layer[0] + 1 ) * layer[1]))
 
    def forward(self, image_LR):
        x = self.features(image_LR)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        weights = []
        for generateW in self.generateWeights:
            weights.append(generateW(x))

        weight_outputs = torch.cat(weights, dim=1)
 
        return weight_outputs, None, x
 
    def hyperAlexWithResidual(layerSizes, pretrained = False, progress = True):
        model = AlexWithResidual(layerSizes)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                                  progress=progress)
            model.load_state_dict(state_dict, strict=False)
        return model