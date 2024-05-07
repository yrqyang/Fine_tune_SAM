import torch
from statistics import mean
from pathlib import Path
from data_loader import DataLoader
from segment_anything import sam_model_registry
import config
from fine_tuning import train_epoch, validate_epoch
import wandb

def main():
    # Initialize wandb
    wandb.init(project="sam_segmentation_project", entity="yrqyang")
    
    # Initialize SAM model
    sam_model = sam_model_registry[config.MODEL_TYPE](checkpoint=config.CHECKPOINT)

    # Use multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs...")
        sam_model = torch.nn.DataParallel(sam_model)
    sam_model.to(config.DEVICE)

    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fn = torch.nn.BCELoss()

    datasets = [
        {'path': config.BASE_PATH, 'new_dataset': False, 'augmentation': False},
        {'path': Path('../dataset/epflml/training'), 'new_dataset': True, 'augmentation': False},
        {'path': config.BASE_PATH, 'new_dataset': False, 'augmentation': True}, # Data augmentation on uavid
        {'path': Path('../dataset/epflml/training'), 'new_dataset': True, 'augmentation': True} # Data augmentation on epflml
    ]

    val_dataset = [
        {'path': config.VAL_PATH, 'new_dataset': False, 'augmentation': False}
    ]

    # Data loaders
    train_loader = DataLoader(datasets, config.DEVICE, sam_model)
    val_loader = DataLoader(val_dataset, config.DEVICE, sam_model)

    # Load data
    train_data = train_loader.load_data()
    val_data = val_loader.load_data()

    # Training and validation loop
    num_epochs = config.NUM_EPOCHS
    for epoch in range(num_epochs):
        train_loss = train_epoch(train_data, sam_model, optimizer, loss_fn)
        print(f'Training Loss after epoch {epoch + 1}: {mean(train_loss)}')

        val_loss = validate_epoch(val_data, sam_model, loss_fn)
        print(f'Validation Loss after epoch {epoch + 1}: {mean(val_loss)}')

        # Log losses to wandb
        wandb.log({"train_loss": mean(train_loss), "val_loss": mean(val_loss)})

    # Save model
    torch.save(sam_model.state_dict(), config.MODEL_SAVE_PATH + "/sam_model_h.pth")

if __name__ == '__main__':
    main()
