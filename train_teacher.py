if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from torch.optim import AdamW #eventualmente posso provare altri ottimizzatori
    from models.teacher import get_teacher
    from utils.dataset import get_dataloaders
    from utils.train_utils import train_one_epoch, evaluate
    from config import *

    train_loader, val_loader, _ = get_dataloaders("data", BATCH_SIZE)

    model = get_teacher(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr = LR, weight_decay=WEIGHT_DECAY)

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)

        print(f"[Teacher] Epoch {epoch}: "
            f"Train Acc {train_acc:.3f} | Val Acc {val_acc:.3f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/teacher.pth") #file binario che contiere i parametri del modello insegnante
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

