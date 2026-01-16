import torch 

def accuracy(outputs, targets):
    _, preds = torch.max(outputs, 1)
    return (preds == targets).float().mean().item() #restituisce top-1 accuracy del batch

# eventualmente posso aggiungere precision, F1....