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

    correct_per_class = torch.zeros(NUM_CLASSES)
    total_per_class = torch.zeros(NUM_CLASSES)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for label, pred in zip(labels, preds):
                label = label.item()
                pred = pred.item()
                total_per_class[label] += 1
                if label == pred:
                    correct_per_class[label] += 1


    avg_time = measure_inference_time(model, test_loader, device)

    print(f"Student Test Accuracy (Top-1): {100 * correct / total:.2f}%")
    print(f"Student inference time: {avg_time * 1000:.2f} ms/img")

     #accuracy per classe
    print("\nAccuracy per class:")
    class_accuracy = correct_per_class / total_per_class

    for i, acc in enumerate(class_accuracy):
        print(f"Class {i}: {acc:.3f}")

    worst_class = torch.argmin(class_accuracy)
    print(
        f"\nWorst class: {worst_class.item()} "
        f"with accuracy {class_accuracy[worst_class]:.3f}"
    )

if __name__ == "__main__":
    main()
