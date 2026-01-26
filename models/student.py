import torch.nn as nn
from torchvision import models

def get_student(num_classes):
    #resnet152 ha 60 milioni di parametri
    model = models.mobilenet_v2(weights="IMAGENET1K_V2") #carico dei pesi pre-addestrati sul database di ImageNet, una sorta di Transfer Learning. Altrimenti avrei valori random
    model.fc = nn.Linear(model.fc.in_features, num_classes) #stanford dogs ha 120 classi, il model.fc addestrato su ImageNet 1000
    return model


# come modelli student prendo shufflenet_v2_x1_5 e mobilenet_v2 perchè hanno un simile numero di parametri (circa 5.5 milioni), altrimenti mobilenet_v3_large/small hanno, rispettivamente, circa 5.5/2.5 milioni di parametri e shufflenet_v2_x2_0 ha circa 7 milioni di parametri