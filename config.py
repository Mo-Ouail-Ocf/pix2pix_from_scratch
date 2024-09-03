from dataclasses import dataclass

@dataclass
class Config:
    DEVICE='cuda'
    LEARNING_RATE = 2e-4
    
    BATCH_SIZE = 8
    NUM_WORKERS=4

    IMAGE_SIZE = 256

    L1_LAMBDA = 100
    LAMBDA_GP=10

    NUM_EPOCHS = 500

    LOAD_MODEL = False
    SAVE_MODEL = True

    BETAS = (0.5,0.999)

