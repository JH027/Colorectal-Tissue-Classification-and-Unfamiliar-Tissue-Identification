import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm


def make_model(num_classes):
    """loads ImageNet-pretrained ResNet-18 and swaps the last layer to output num_classes scores"""
    """take in batch of images, output batch number of rows and 8 columns --> each row holds one image's 8 scores --> highest is the prediction"""
    #loads weights pretrained on imgs --> gives basically head start to detecting edges textures shape --> summarize as numbers that capture key things abt img
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    #reads 512 numbers from the img and decides which class it is
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """one pass over training data, updates weights, returns avg loss and accuracy"""
    model.train()
    total_loss, correct, count = 0.0, 0, 0
    bar = tqdm(loader, unit="batch", leave=False)
    for x, y in bar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (out.argmax(dim=1) == y).sum().item()
        count += len(y)
    return total_loss / count, correct / count

def evaluate(model, loader, loss_fn, device):
    """one pass over validation data with no weight updates, returns avg loss and accuracy"""
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            total_loss += loss.item() * len(y)
            correct += (out.argmax(dim=1) == y).sum().item()
            count += len(y)
    return total_loss / count, correct / count