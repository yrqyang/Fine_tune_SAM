import torch
import torch.nn.functional as F
from torch.nn.functional import threshold, normalize
from tqdm import tqdm
import torchvision.transforms as transforms

def train_epoch(data_loader, model, optimizer, loss_fn, device):
    model.train()  # Ensure the model is in training mode
    epoch_losses = []

    for data in tqdm(data_loader):
        input_images = data['image'].squeeze(1).to(device)
        gt_masks = data['mask'].squeeze(1).to(device)
        input_sizes = data['input_size']
        original_image_sizes = data['original_image_size']

        optimizer.zero_grad()
        loss = process_batch(input_images, gt_masks, input_sizes, original_image_sizes, model, loss_fn, device)
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

    return epoch_losses

# def validate_epoch(data, model, loss_fn):
def validate_epoch(data_loader, model, loss_fn, device):
    model.eval()  # Ensure the model is in evaluation mode
    epoch_losses = []

    with torch.no_grad():
        for data in data_loader:
            input_images = data['image'].squeeze(1).to(device)
            gt_masks = data['mask'].squeeze(1).to(device)
            input_sizes = data['input_size']
            original_image_sizes = data['original_image_size']

            loss = process_batch(input_images, gt_masks, input_sizes, original_image_sizes, model, loss_fn, device)
            epoch_losses.append(loss.item())

    return epoch_losses

# def process_data(data_point, model, optimizer, loss_fn, train):
def process_batch(input_images, gt_masks, input_sizes, original_image_sizes, model, loss_fn, device):
    with torch.no_grad():
        image_embedding = model.image_encoder(input_images)

        #sparse_embeddings, dense_embeddings = model.prompt_encoder(points=None, boxes=None, masks=gt_mask)
        sparse_embeddings, dense_embeddings = model.prompt_encoder(points=None, boxes=None, masks=None)
        dense_embeddings_resized = F.interpolate(dense_embeddings, size=(64, 64), mode='bilinear', align_corners=False)

    low_res_masks, _ = model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings_resized,
        multimask_output=False
    )

    # YT : BATCH : batchify the postprocessing
    #upscaled_masks = torch.stack([model.postprocess_masks(low_res_masks[i].unsqueeze(0), [_[i] for _ in input_sizes], [_[i] for _ in original_image_sizes]).squeeze(0) for i in range(low_res_masks.shape[0])])
    
    upscaled_masks = model.postprocess_masks(low_res_masks, input_sizes, original_image_sizes)
    predicted_masks = torch.sigmoid(upscaled_masks)

    # YT : BATCH : GT Masks Size uniformization for batch stacking problem (pixel differences due to the interpolation)
    #gt_masks = torch.stack([transforms.Resize(torch.stack(original_image_sizes)[:, i].tolist()).forward(gt_masks[i]) for i in range(gt_masks.shape[0])])

    # Clip the masks between 0 and 1 before passing it to loss to avoid error : Assertion `target_val >= zero && target_val <= one` failed.
    predicted_masks, gt_masks = torch.clamp(predicted_masks, 0, 1), torch.clamp(gt_masks, 0, 1)
    loss = loss_fn(predicted_masks.to(device).float(), gt_masks.to(device).float())

    return loss
