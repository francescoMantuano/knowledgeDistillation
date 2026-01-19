import os
import random
import shutil

# ===== CONFIG =====
SOURCE_DIR = "images\Images"   # cartella originale
TARGET_DIR = "datasets"               # train/val/test
TRAIN_RATIO = 0.8 #80%
VAL_RATIO = 0.1 #10%
TEST_RATIO = 0.1 #10%
SEED = 42
# ==================

random.seed(SEED)

# crea cartelle target
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(TARGET_DIR, split), exist_ok=True)

# ciclo su ogni classe
for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * TRAIN_RATIO)
    n_val   = int(n_total * VAL_RATIO)

    splits = {
        "train": images[:n_train],
        "val":   images[n_train:n_train + n_val],
        "test":  images[n_train + n_val:]
    }

    for split, split_images in splits.items():
        split_class_dir = os.path.join(TARGET_DIR, split, class_name)
        os.makedirs(split_class_dir, exist_ok=True)

        for img in split_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(split_class_dir, img)
            shutil.copy(src, dst)

    print(f"{class_name}: {n_train} train / {n_val} val / {n_total - n_train - n_val} test")

print("\n Dataset splittato correttamente (80/10/10) con classi bilanciate.")
