import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os.path
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
from io import BytesIO

class LambdaLR:
    def __init__(self, n_epochs, decay_start_epoch):
        self.n_epochs = n_epochs #total epochs
        self.decay_start_epoch = decay_start_epoch # Epoch starting to reduce learning rate

    def step(self, epoch):
        return 1.0 - max(0, epoch - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

def randomJPEGcompression(image):
    pb = random.random() 
    if pb <= 0.9:
        qf = random.randrange(10, 100,10) #start,stop,step
        outputIoStream = BytesIO()
        image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
        outputIoStream.seek(0)
        return Image.open(outputIoStream)
    else : 
        return image

def randomRescale(image):
    pb = random.random()
    #print("pb", pb)
    if pb <= 0.9:
        rescale_step = random.randrange(50)
        min = 0.25
        max = 4
        rescale_ratio = ((max - min)/50) * rescale_step+1 + min
        transform = transforms.Resize((int(224*rescale_ratio),int(224*rescale_ratio)))
        transform2 = transforms.Resize((224,224))
        return transform2(transform(image))
    else: 
        return image


def randomRotation(image):
    angle = random.randrange(-180,180,90) #start,stop,step
    return transforms.functional.rotate(image,angle)

transformation_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
          
transformation_train = transforms.Compose(
    [transforms.ToTensor(),
     transforms.RandomHorizontalFlip(p=0.9),
     transforms.RandomVerticalFlip(p=0.9),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transformation_Forchheim_train = transforms.Compose(
    [
     transforms.RandomHorizontalFlip(p=0.9),
     transforms.RandomVerticalFlip(p=0.9),
     transforms.Lambda(randomRotation),
     transforms.Lambda(randomRescale),
     transforms.Lambda(randomJPEGcompression),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     ])

def imshow(img,title):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()
