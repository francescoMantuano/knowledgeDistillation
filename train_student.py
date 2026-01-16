import torch
import torch.nn as nn
from torch.optim import AdamW
from models.student import get_student
from utils.dataset import get_dataloaders
from utils.train_utils import train_one_epoch, evaluate
from config import *

train_loader, val_loader, _ = get_dataloaders("data", BATCH_SIZE)

model = get_student(NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

best_loss = float("inf")

for epoch in range(NUM_EPOCHS):
    
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)

    print(f"[Student] Epoch {epoch}: "
          f"Train Acc {train_acc:.3f} | Val Acc {val_acc:.3f}")

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "student.pth") #file binario che contiere i parametri del modello studente
