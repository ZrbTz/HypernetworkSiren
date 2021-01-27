from . import *

###############################################################################################
#Modified version of the alexnet, this is the one the worked the best#

class HyperMetaAlexNet(nn.Module):
    def __init__(self, layerSizes):
        super(HyperMetaAlexNet, self).__init__()
 
        self.layerSizes = layerSizes

        #Feature Extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3 , stride=2, padding=1),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((3,3))

        #Initialization of convolutional layers with Xavier Uniform
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

        #FC layers, a pair for each target layer (weights and bias) 
        self.generateWeights = nn.ModuleList()
        self.generateBiases = nn.ModuleList()

        #Initialization of FC layers
        for i, layer in enumerate(self.layerSizes):
            w_linear = nn.Linear(256 * 3 * 3, layer[0] * layer[1])
            b_linear = nn.Linear(256 * 3 * 3, layer[1])

            with torch.no_grad():
                w_linear.weight.uniform_(-1 / (256 * 3 * 3), 1 / (256 * 3 * 3))
                b_linear.weight.uniform_(-1 / (256 * 3 * 3), 1 / (256 * 3 * 3))

            self.generateWeights.append(w_linear)
            self.generateBiases.append(b_linear)
 
    def forward(self, image_LR):
        x = self.features(image_LR)
        #x = self.avgpool(x)
        x = torch.flatten(x, 1)

        #FC layers
        weights = []
        biases = []
        for i, generateW in enumerate(self.generateWeights):
            weights.append(generateW(x))
            biases.append(self.generateBiases[i](x))

        weight_outputs = torch.cat(weights, dim=1)
        biases_outputs = torch.cat(biases, dim=1)
 
        return weight_outputs, biases_outputs, x
 
    def hyperMetaAlexNet(layerSizes, pretrained = True, progress = True):
        model = HyperMetaAlexNet(layerSizes)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls['alexnet'], progress=progress)
            model.load_state_dict(state_dict, strict=False)
        return model

###############################################################################################
#BASE ALEXNET#

class HyperBaseAlexNet(nn.Module):
    def __init__(self, layerSizes):
        super(HyperBaseAlexNet, self).__init__()
 
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
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.generateWeights = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.totalNumberOfWeights + self.totalNumberOfBiases),
        )
 
    def forward(self, image_LR):
        x = self.features(image_LR)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        weight_outputs = self.generateWeights(x)
 
        return weight_outputs
 
    def hyperBaseAlexNet(layerSizes, pretrained = True, progress = True):
        model = HyperBaseAlexNet(layerSizes)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                                  progress=progress)
            model.load_state_dict(state_dict, strict=False)
        return model

###############################################################################################

class HyperBaseAlexNetFL(nn.Module):
    def __init__(self, layerSizes):
        super(HyperBaseAlexNetFL, self).__init__()
 
        self.layerSizes = layerSizes

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
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
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
 
        return weight_outputs
 
    def hyperBaseAlexNetFL(layerSizes, pretrained = True, progress = True):
        model = HyperBaseAlexNetFL(layerSizes)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls['alexnet'], progress=progress)
            model.load_state_dict(state_dict, strict=False)
        return model