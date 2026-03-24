import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 120
BATCH_SIZE = 64
#conviene fare un batch size sperato per teacher/student?

NUM_EPOCHS = 200
LR = 3e-4 #1e-4 prima

WEIGHT_DECAY = 1e-4 #dropout per regolarizzare

IMAGE_SIZE = 224

KD_TEMPERATURE = 3 #eventualmente modificabile

KD_ALPHA = 0.4 #kd loss
KD_BETA = 0.55  #feature loss
KD_GAMMA = 0.03 #ce loss
KD_DELTA = 0.01 #relationship loss

DROPOUT_RATE = 0.3

REL_WARMUP = 10 #warmup per la relationship loss siccome rumorosa all'inizio del training
PATIENCE = 15 #early stopping per regolarizzare

    