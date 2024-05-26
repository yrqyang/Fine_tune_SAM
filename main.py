import torch
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from statistics import mean
from pathlib import Path
# from data_loader import DataLoader
from data_loader import CustomDataset
from segment_anything import sam_model_registry
import config
from fine_tuning import train_epoch, validate_epoch

import wandb


def main():
    wandb.init(project="sam_segmentation_project", entity="yrqyang")
    # Initialize SAM model
    sam_model = sam_model_registry[config.MODEL_TYPE](checkpoint=config.CHECKPOINT)

    # Freeze image and prompt encoder -> gradients computed only for mask decoder
    for name, param in sam_model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)
    
    sam_model.to(config.DEVICE)

    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fn = torch.nn.BCELoss()

    datasets = [
        {'path': config.BASE_PATH, 'new_dataset': False, 'augmentation': False},
        {'path': Path('/home/student/RoadSegmentation/RoadSegmentation/dataset/epflml/training'), 'new_dataset': True, 'augmentation': False},
        {'path': config.BASE_PATH, 'new_dataset': False, 'augmentation': True}, # Data augmentation on uavid
        {'path': Path('/home/student/RoadSegmentation/RoadSegmentation/dataset/epflml/training'), 'new_dataset': True, 'augmentation': True} # Data augmentation on epflml
    ]

    val_dataset = [
        {'path': config.VAL_PATH, 'new_dataset': False, 'augmentation': False}
    ]

    # Data Set
    train_dataset = CustomDataset(datasets, config.DEVICE, sam_model)
    val_dataset = CustomDataset(val_dataset, config.DEVICE, sam_model)

    # Data Loader
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Training and validation loop
    num_epochs = config.NUM_EPOCHS

    for epoch in range(num_epochs):
        # train_losses = []
        # for batch_data in create_batches(train_data, batch_size):
        #     batch_data = batch_data.to(config.DEVICE)
        #     train_loss = train_epoch(batch_data, sam_model, optimizer, loss_fn)
        #     train_losses.extend(train_loss)
        train_losses = train_epoch(train_loader, sam_model, optimizer, loss_fn, config.DEVICE)
        
        # if rank == 0:  # Only the master process
        print(f'Training Loss after epoch {epoch + 1}: {mean(train_losses)}')
        # print(f'Training Loss after epoch {epoch + 1}: {mean(train_loss)}')

        # val_losses = []
        # for batch_data in create_batches(val_data, batch_size):
        #     val_loss = validate_epoch(batch_data, sam_model, loss_fn)
        #     val_losses.extend(val_loss)

        val_losses = validate_epoch(val_loader, sam_model, loss_fn, config.DEVICE)
        
        print(f'Validation Loss after epoch {epoch + 1}: {mean(val_losses)}')    
        
        # Log losses to wandb
        wandb.log({"train_loss": mean(train_losses), "val_loss": mean(val_losses)})


    # Save model
    torch.save(sam_model.state_dict(), config.MODEL_SAVE_PATH + "/sam_model_h.pth")


if __name__ == '__main__':
    main()