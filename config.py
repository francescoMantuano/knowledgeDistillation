import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 120
BATCH_SIZE = 64
#conviene fare un batch size sperato per teacher/student?

NUM_EPOCHS = 200
LR = 3e-4 #1e-4 prima

WEIGHT_DECAY = 1e-4 #per regolarizzare

IMAGE_SIZE = 224

KD_TEMPERATURE = 4 #eventualmente modificabile
KD_ALPHA = 0.2 #eventualmente modificabile

PATIENCE = 15 #early stopping per regolarizzare

DROPOUT_RATE = 0.3 # per bloccare alcune funzioni di attivazione
    