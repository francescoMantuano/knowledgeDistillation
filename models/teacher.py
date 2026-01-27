import torch.nn as nn
from torchvision import models

def get_teacher(num_classes):
    #resnet152 ha 60 milioni di parametri
    model = models.resnet152(weights="IMAGENET1K_V2") #carico dei pesi pre-addestrati sul database di ImageNet, una sorta di Transfer Learning. Altrimenti avrei valori random
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model