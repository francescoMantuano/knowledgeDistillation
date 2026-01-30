if __name__ == "__main__":    
    import torch
    from torch.optim import AdamW
    from models.teacher import get_teacher
    from models.student import get_student
    from utils.dataset import get_dataloaders
    from utils.losses import distillation_loss, feature_distillation_loss
    from utils.metrics import accuracy
    from utils.train_utils import count_params
    from config import *
    import time
    import os
    from utils.hooks import FeatureHook
    from utils.projection import FeatureProjector
    from utils.model_utils import get_feature_channels, get_kd_feature_layer

    best_val_loss = float("inf")
    patience_counter = 0
    actual_epochs = 0

    train_loader, val_loader, _ = get_dataloaders("datasets", BATCH_SIZE)

    teacher = get_teacher(NUM_CLASSES).to(DEVICE)
    teacher.load_state_dict(torch.load("checkpoints/teacher.pth"))
    teacher.eval()

    student = get_student(NUM_CLASSES).to(DEVICE)

    #estraggo il layer apposito a cui agganciarmi con l'hook in base al modello
    teacher_layer = get_kd_feature_layer(teacher)
    student_layer = get_kd_feature_layer(student)

    #hooks da agganciare a teacher e student
    teacher_hook = FeatureHook(teacher_layer)
    student_hook = FeatureHook(student_layer)

    #estrazione canali da student e teacher
    teacher_channels = get_feature_channels(teacher, teacher_hook, DEVICE)
    student_channels = get_feature_channels(student, student_hook, DEVICE)

    projector = FeatureProjector(in_channels=student_channels, out_channels=teacher_channels).to(DEVICE)
    
    #l'ottimizzatore va aggiornato anche tenendo conto del projector
    optimizer = AdamW(list(student.parameters()) + list(projector.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        actual_epochs += 1
        student.train()
        epoch_loss = 0.0
        epoch_acc = 0.0

        # loop di training
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            teacher_hook.clear()
            student_hook.clear()

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = teacher(x)
                teacher_feat = teacher_hook.features
            teacher_feat = teacher_hook.features.detach() #per extra sicurezza, il detach è già implicito se uso with torch.no_grad(): teacher(x)

            student_logits = student(x)
            student_feat = student_hook.features

            #proiezione a causa delle diverse dimensioni di teacher e student
            student_feat_proj = projector(student_feat)

            loss_logits = distillation_loss(student_logits, teacher_logits, y, KD_TEMPERATURE, KD_ALPHA, KD_GAMMA)
            loss_feat = feature_distillation_loss(student_feat_proj, teacher_feat)
            loss = loss_logits + KD_BETA * loss_feat
            
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

                teacher_logits = teacher(x_val)
                teacher_feat = teacher_hook.features
                student_logits = student(x_val)
                student_feat = student_hook.features
                student_feat_proj = projector(student_feat)

                loss_logits = distillation_loss(student_logits, teacher_logits, y, KD_TEMPERATURE, KD_ALPHA, KD_GAMMA)
                loss_feat = feature_distillation_loss(student_feat_proj, teacher_feat)
                loss = loss_logits + KD_BETA * loss_feat
            

                val_loss += loss.item()
                val_acc += accuracy(student_logits, y_val)
            
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        print(f"[Student KD] Epoch {epoch}: Train Acc {epoch_acc:.3f} | Val Acc {val_acc:.3f} | Train Loss {epoch_loss:.3f} | Val Loss {val_loss:.3f} | Logits Loss {loss_logits:.3f} | Feat Loss {loss_feat:.3f}")

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
    avg_epoch_time = total_time / actual_epochs

    size_student_kd = os.path.getsize("checkpoints/student_kd.pth") / (1024**2)

    print(f"\nTotal training time: {total_time/60:.2f} minutes")
    print(f"Avg time per epoch: {avg_epoch_time:.2f} seconds")
    print(f"Student+KD size: {size_student_kd:2f} MB")
    print("Student+KD params: ", count_params(student))
