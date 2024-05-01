from pathlib import Path
import torch

# Path configurations
BASE_PATH = Path('../dataset/uavid/uavid_train')
VAL_PATH = Path('../dataset/uavid/uavid_val')
TEST_PATH = Path('../dataset/uavid/uavid_test')
MODEL_SAVE_PATH = "../models"

# Model configurations
MODEL_TYPE = 'vit_h' # update to .h type pretrained model
CHECKPOINT = '../sam_vit_h_4b8939.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training configurations
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0
NUM_EPOCHS = 100
