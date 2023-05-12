import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pretrainedmodels
from efficientnet_pytorch import EfficientNet

class ConvNet(nn.Module):
  #convolution size formula (W - F + 2*P)/S + 1
    def __init__(self,number_of_class=2): #initialize function names
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), #(256-3+2)/1+1 = 256
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64, kernel_size=17, stride=2, padding=0), #(256-34+0)/1+1 = 112
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), 
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=18432, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=nbr_of_class, bias=True),
          )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        #x = x.view(-1, 512 * 6 * 6)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    
class EfficientNet_b0_vanila(nn.Module):
    def __init__(self,nbr_of_class):
        super(EfficientNet_b0, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')     
        self.classifier_layer = nn.Sequential(
                    nn.Linear(1280 , 512),
                    nn.BatchNorm1d(512),
                    nn.Dropout(0.2),
                    nn.Linear(512 , 256),
                    nn.Linear(256 , nbr_of_class))
        
    def forward(self, inputs):
        x = self.model.extract_features(inputs)
        x = self.model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.model._dropout(x)
        x = self.classifier_layer(x)
        return x
      
      
class EfficientNet_b0(nn.Module):
    def __init__(self, nbr_of_class):
        super(EfficientNet_b0, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        #self.model = EfficientNet.from_name('efficientnet-b0')
        #self.model.load_state_dict(torch.load("/linkhome/rech/genwnn01/uru89tg/.cache/torch/hub/checkpoints/efficientnet-b0-355c32eb.pth"), strict=False)
        # Remove final classification layer
        self.model._fc = nn.Identity()

        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier_layer = nn.Sequential(
            nn.Linear(self.model._conv_head.out_channels, nbr_of_class),
            nn.Softmax(dim=1)
        )
        # Initialize the weights of the new classification layer with Glorot uniform initialization
        nn.init.xavier_uniform_(self.classifier_layer[0].weight)

    def forward(self, inputs):
        x = self.model.extract_features(inputs)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.classifier_layer(x)
        return x
      
class EfficientNet_b5(nn.Module):
    def __init__(self,nbr_of_class):
        super(EfficientNet_b5, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b5')     
        self.classifier_layer = nn.Sequential(
                    nn.Linear(2048 , 512),
                    nn.BatchNorm1d(512),
                    nn.Dropout(0.2),
                    nn.Linear(512 , 256),
                    nn.Linear(256 , nbr_of_class))
        
    def forward(self, inputs):
        x = self.model.extract_features(inputs)
        x = self.model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.model._dropout(x)
        x = self.classifier_layer(x)
        return x
      
class ResNet18(nn.Module): 
      def __init__(self,nbr_of_class):
          super(ResNet18, self).__init__()
          self.model = models.resnet18(pretrained=True)
          num_ftrs = self.model.fc.in_features #resnet
          self.model.fc = nn.Linear(num_ftrs, nbr_of_class) #resnet
        
      def forward(self, inputs):
          x = self.model(inputs)
          return x
        
      
