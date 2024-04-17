from pathlib import Path
import torch

# Path configurations
BASE_PATH = Path('../dataset/uavid_train')
VAL_PATH = Path('../dataset/uavid_val')
TEST_PATH = Path('../dataset/uavid_test')
MODEL_SAVE_PATH = "./models"

# Model configurations
MODEL_TYPE = 'vit_b'
CHECKPOINT = 'sam_vit_b_01ec64.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training configurations
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0
NUM_EPOCHS = 100
