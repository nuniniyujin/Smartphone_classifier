import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os.path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np

class LambdaLR:
    def __init__(self, n_epochs, decay_start_epoch):
        self.n_epochs = n_epochs #total epochs
        self.decay_start_epoch = decay_start_epoch # Epoch starting to reduce learning rate

    def step(self, epoch):
        return 1.0 - max(0, epoch - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

      
transformation = transforms.Compose(
    [transforms.ToTensor(),
     transforms.RandomHorizontalFlip(p=0.9),
     transforms.RandomVerticalFlip(p=0.9),
     transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.9),
     #A.ImageCompression(quality_lower=10, quality_upper=100),
     #A.Downscale(scale_min = 0.5, scale_max = 4),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def imshow(img,title):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()
