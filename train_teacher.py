if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from torch.optim import AdamW #eventualmente posso provare altri ottimizzatori
    from models.teacher import get_teacher
    from utils.dataset import get_dataloaders
    from utils.train_utils import train_one_epoch, evaluate, count_params
    from config import *
    import time
    import os

    train_loader, val_loader, _ = get_dataloaders("datasets", BATCH_SIZE)

    model = get_teacher(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr = LR, weight_decay=WEIGHT_DECAY)

    best_loss = float("inf")
    patience_counter = 0
    actual_epochs = 0

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        actual_epochs += 1
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)

        print(f"[Teacher] Epoch {epoch}: "
            f"Train Acc {train_acc:.3f} | Val Acc {val_acc:.3f} | Train Loss {train_loss:.3f}  | Val Loss {val_loss:.3f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/teacher.pth") #file binario che contiere i parametri del modello insegnante
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    #per evitare sottostima del tempo in ambiente cuda        
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()

    size_teacher = os.path.getsize("checkpoints/teacher.pth") / (1024**2)

    total_time = end_time - start_time
    avg_epoch_time = total_time / actual_epochs

    print(f"\nTotal training time: {total_time/60:.2f} minutes")
    print(f"Avg time per epoch: {avg_epoch_time:.2f} seconds")
    print(f"Teacher size: {size_teacher:2f} MB")
    print("Teacher params:", count_params(model))

