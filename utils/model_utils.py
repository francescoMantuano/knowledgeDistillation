import torch
from config import *
from utils.hooks import FeatureHook

#nel caso stanford dogs IMAGE SIZE = 224*224
def get_feature_channels(model, layer, device, input_size=(1,3,IMAGE_SIZE,IMAGE_SIZE)):
    #metto il modello in eval a causa di BatchNorm e Dropout
    model.eval()
    
    #creo un hook per il layer scelto
    hook = FeatureHook(layer)

    with torch.no_grad():
        dummy = torch.zeros(input_size).to(device)
        _ = model(dummy)
    
    channels = hook.features.shape[1]
    hook.close()

    return channels

def get_kd_feature_layer(model):
    name = model.__class__.__name__.lower()

    if "resnet" in name:
        return model.layer4
    
    elif "shufflenet" in name:
        return model.stage4
    
    elif "mobilenet" in name:
        return model.features[-1]
    
    else:
        raise ValueError(f"Unsupported model for KD: {name}")
    
def geometric_warmup(epoch, max_value, warmup_epochs, power):
    if epoch >= warmup_epochs:
        return max_value
    return max_value * (epoch / warmup_epochs) ** power

def linear_warmup(epoch, warmup_epochs, target_value):
    if warmup_epochs == 0:
        return target_value

    if epoch >= warmup_epochs:
        return target_value

    return target_value * (epoch / warmup_epochs)