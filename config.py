import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 120
BATCH_SIZE = 64
#conviene fare un batch size sperato per teacher/student?
NUM_EPOCHS = 30
LR = 3e-4
WEIGHT_DECAY = 1e-4 #dropout per regolarizzare

IMAGE_SIZE = 224

KD_TEMPERATURE = 4.0 #eventualmente modificabile
KD_ALPHA = 0.7 #eventualmente modificabile

PATIENCE = 50 #early stopping per regolarizzare

    