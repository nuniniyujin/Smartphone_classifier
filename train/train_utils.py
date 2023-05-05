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
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transformation2 = A.Compose([
    A.RandomCrop(width=224, height=224),
    A.augmentations.geometric.rotate.RandomRotate90(always_apply=False, p=1.0),
    A.augmentations.geometric.transforms.HorizontalFlip(0.5),
    A.augmentations.geometric.transforms.VerticalFlip(0.5),
    A.augmentations.transforms.ImageCompression(quality_lower=10, quality_upper=90, p=0.9),
    A.augmentations.transforms.Downscale(scale_limit=(0.25,4), p=0.9),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),ToTensorV2()
])

def imshow(img,title):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()
