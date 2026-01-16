import torch.nn as nn
from torchvision import models

def get_teacher(num_classes):
    model = models.resnet50(weights="IMAGENET1K_V1") #carico dei pesi pre-addestrati sul database di ImageNet, una sorta di Transfer Learning. Altrimenti avrei valori random
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model