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

KD_ALPHA = 0.3 #kd loss
KD_BETA = 0.05  #feature loss
KD_GAMMA = 0.5 #ce loss
KD_DELTA = 0.02 #relationship loss

DROPOUT_RATE = 0.3

REL_WARMUP = 50 #warmup per la relationship loss siccome rumorosa all'inizio del training
POWER = 3 #potenza dell progressione geometrica
PATIENCE = 15 #early stopping per regolarizzare

    