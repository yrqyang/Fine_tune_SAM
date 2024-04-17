import torch
import numpy as np

def rgb_to_binary_mask(label):
    # Target color for the class Road
    target_color = (128, 64, 128)
    
    # Initialize the binary mask with zeros
    binary_mask = np.zeros((label.shape[0], label.shape[1]), dtype=np.int32)
    
    # Set pixels that match the target color to 1
    matches = (label == target_color).all(axis=-1)
    binary_mask[matches] = 1

    return binary_mask
