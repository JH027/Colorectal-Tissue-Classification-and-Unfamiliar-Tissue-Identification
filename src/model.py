import torch.nn as nn
from torchvision import models

def make_model(num_classes):
    """loads ImageNet-pretrained ResNet-18 and swaps the last layer to output num_classes scores"""
    """take in batch of images, output batch number of rows and 8 columns --> each row holds one image's 8 scores --> highest is the prediction"""
    #loads weights pretrained on imgs --> gives basically head start to detecting edges textures shape --> summarize as numbers that capture key things abt img
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    #reads 512 numbers from the img and decides which class it is
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model