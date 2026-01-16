import torch.nn as nn
from torchvision import models

def get_student(num_classes):
    model = models.resnet18(weights="IMAGENET1K_V1") #carico dei pesi pre-addestrati sul database di ImageNet, una sorta di Transfer Learning. Altrimenti avrei valori random
    model.fc = nn.Linear(model.fc.in_features, num_classes) #stanford dogs ha 120 classi, il model.fc addestrato su ImageNet 1000
    return model