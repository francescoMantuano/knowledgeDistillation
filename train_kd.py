if __name__ == "__main__":    
    import torch
    from torch.optim import AdamW
    from models.teacher import get_teacher
    from models.student import get_student
    from utils.dataset import get_dataloaders
    from utils.losses import distillation_loss
    from utils.metrics import accuracy
    from config import *

    train_loader, val_loader, _ = get_dataloaders("data", BATCH_SIZE)

    teacher = get_teacher(NUM_CLASSES).to(DEVICE)
    teacher.load_state_dict(torch.load("checkpoints/teacher.pth"))
    teacher.eval()

    student = get_student(NUM_CLASSES).to(DEVICE)
    optimizer = AdamW(student.parameters(), lr=LR)

    for epoch in range(NUM_EPOCHS):
        student.train()
        total_acc = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            with torch.no_grad():
                teacher_logits = teacher(x)

            student_logits = student(x)

            loss = distillation_loss(student_logits, teacher_logits, y, KD_TEMPERATURE, KD_ALPHA)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_acc += accuracy(student_logits, y)

        print(f"[Student KD] Epoch {epoch}: Acc {total_acc / len(train_loader):.3f}")
        
    torch.save(student.state_dict(), "checkpoints/student_kd.pth") #file binario che contiere i parametri del modello studente allenato con kd

