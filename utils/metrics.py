import torch 
import time

def accuracy(outputs, targets):
    _, preds = torch.max(outputs, 1)
    return (preds == targets).float().mean().item() #restituisce top-1 accuracy del batch

# eventualmente posso aggiungere precision, F1....

def measure_inference_time(model, dataloader, device):
    model.eval()
    model.to(device)

    total_time = 0.0
    total_images = 0

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)

            start = time.perf_counter()
            _ = model(x)
            end = time.perf_counter()

            total_time += (end - start)
            total_images += x.size(0)

    avg_time_per_image = total_time / total_images
    return avg_time_per_image