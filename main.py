import torch
from statistics import mean
from data_loader import DataLoader
from segment_anything import sam_model_registry
import config
from fine_tuning import train_epoch, validate_epoch

def main():
    # Initialize SAM model
    sam_model = sam_model_registry[config.MODEL_TYPE](checkpoint=config.CHECKPOINT)
    sam_model.to(config.DEVICE)

    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fn = torch.nn.BCELoss()

    # Data loaders
    train_loader = DataLoader(config.BASE_PATH, config.DEVICE, sam_model)
    val_loader = DataLoader(config.VAL_PATH, config.DEVICE, sam_model)
    test_loader = DataLoader(config.TEST_PATH, config.DEVICE, sam_model)

    # Load data
    train_data = train_loader.load_data()
    val_data = val_loader.load_data()
    test_data = test_loader.load_data()

    # Training and validation loop
    num_epochs = config.NUM_EPOCHS
    for epoch in range(num_epochs):
        train_loss = train_epoch(train_data, sam_model, optimizer, loss_fn)
        print(f'Training Loss after epoch {epoch + 1}: {mean(train_loss)}')

        val_loss = validate_epoch(val_data, sam_model, loss_fn)
        print(f'Validation Loss after epoch {epoch + 1}: {mean(val_loss)}')

    # Testing before saving the model
    test_loss = validate_epoch(test_data, sam_model, loss_fn)
    print(f'Test Loss: {mean(test_loss)}')

    # Save model
    torch.save(sam_model.state_dict(), config.MODEL_SAVE_PATH + "/sam_model.pth")

if __name__ == '__main__':
    main()
