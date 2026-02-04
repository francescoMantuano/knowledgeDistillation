import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 120
BATCH_SIZE = 64
#conviene fare un batch size sperato per teacher/student?

NUM_EPOCHS = 200
LR = 3e-4 #1e-4 prima

WEIGHT_DECAY = 1e-4 #dropout per regolarizzare

IMAGE_SIZE = 224

KD_TEMPERATURE = 4 #eventualmente modificabile

KD_ALPHA = 0.7 #kd loss
KD_BETA = 0.2  #feature loss
KD_GAMMA = 2  #ce loss
KD_DELTA = 0.1 #relationship loss

DROPOUT_RATE = 0.3

PATIENCE = 15 #early stopping per regolarizzare

    