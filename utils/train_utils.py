import torch
from utils.metrics import accuracy

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_acc = 0,0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad() #per evitare che i gradienti si accumulino, altrimenti gradiente nuovo = gradiente vecchio + nuovo
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy(out, y)

    return total_loss/ len(loader), total_acc / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc = 0,0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)

        total_loss += loss.item()
        total_acc += accuracy(out, y)

    return total_loss / len(loader), total_acc / len(loader)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
