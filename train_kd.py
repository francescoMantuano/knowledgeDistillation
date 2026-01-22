if __name__ == "__main__":    
    import torch
    from torch.optim import AdamW
    from models.teacher import get_teacher
    from models.student import get_student
    from utils.dataset import get_dataloaders
    from utils.losses import distillation_loss
    from utils.metrics import accuracy
    from config import *
    import time

    best_val_loss = float("inf")
    patience_counter = 0
    train_loader, val_loader, _ = get_dataloaders("datasets", BATCH_SIZE)

    teacher = get_teacher(NUM_CLASSES).to(DEVICE)
    teacher.load_state_dict(torch.load("checkpoints/teacher.pth"))
    teacher.eval()

    student = get_student(NUM_CLASSES).to(DEVICE)
    optimizer = AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        student.train()
        epoch_loss = 0.0
        epoch_acc = 0.0

        # loop di training
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            with torch.no_grad():
                teacher_logits = teacher(x)

            student_logits = student(x)

            loss = distillation_loss(student_logits, teacher_logits, y, KD_TEMPERATURE, KD_ALPHA)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += accuracy(student_logits, y)

        epoch_loss /= len(train_loader)
        epoch_acc /= len(train_loader)

        #early stopping per kd
        student.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(DEVICE), y_val.to(DEVICE)

                student_logits = student(x_val)
                teacher_logits = teacher(x_val)

                loss = distillation_loss(student_logits, teacher_logits, y_val, KD_TEMPERATURE, KD_ALPHA)

                val_loss += loss.item()
                val_acc += accuracy(student_logits, y_val)
            
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        print(f"[Student KD] Epoch {epoch}: Train Acc {epoch_acc:.3f} | Val Acc {val_acc:.3f} | Train Loss {epoch_loss:.3f} | Val Loss {val_loss:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # salva il modello migliore
            torch.save(student.state_dict(), "checkpoints/student_kd.pth") #file binario che contiere i parametri del modello studente allenato con kd
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        student.train()  # torna in modalità train per la prossima epoca

    #per evitare sottostima del tempo in ambiente cuda        
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()

    total_time = end_time - start_time
    avg_epoch_time = total_time / NUM_EPOCHS

    print(f"\nTotal training time: {total_time/60:.2f} minutes")
    print(f"Avg time per epoch: {avg_epoch_time:.2f} seconds")

