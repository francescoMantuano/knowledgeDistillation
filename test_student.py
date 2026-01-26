import torch
from config import DEVICE, NUM_CLASSES, BATCH_SIZE
from utils.dataset import get_dataloaders
from models.student import get_student  # oppure import Student se è una classe
from utils.metrics import measure_inference_time

CHECKPOINT = "checkpoints/student.pth"

def main():
    device = DEVICE

    _, _, test_loader = get_dataloaders(batch_size=BATCH_SIZE, data_root="datasets")

    model = get_student(num_classes=NUM_CLASSES)  # usa la tua funzione
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)


    avg_time = measure_inference_time(model, test_loader, device)

    print(f"Student Test Accuracy (Top-1): {100 * correct / total:.2f}%")
    print(f"Student inference time: {avg_time * 1000:.2f} ms/img")

if __name__ == "__main__":
    main()
