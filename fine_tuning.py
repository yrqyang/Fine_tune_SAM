import torch
import torch.nn.functional as F
from torch.nn.functional import threshold, normalize
from tqdm import tqdm
from statistics import mean

def train_epoch(data, model, optimizer, loss_fn):
    model.train()  # Ensure the model is in training mode
    epoch_losses = []
    for key in tqdm(data.keys()):
        data_point = data[key]
        loss = process_data(data_point, model, optimizer, loss_fn, train=True)
        epoch_losses.append(loss)
    return epoch_losses

def validate_epoch(data, model, loss_fn):
    model.eval()  # Ensure the model is in evaluation mode
    epoch_losses = []
    for key in data.keys():
        data_point = data[key]
        loss = process_data(data_point, model, None, loss_fn, train=False)
        epoch_losses.append(loss)
    return epoch_losses

def process_data(data_point, model, optimizer, loss_fn, train):
    input_image = data_point['image']
    gt_mask = data_point['mask']
    input_size = data_point['input_size']
    original_image_size = data_point['original_image_size']

    image_embedding = model.image_encoder(input_image)

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

    upscaled_masks = model.postprocess_masks(low_res_masks, input_size, original_image_size)
    predicted_masks = torch.sigmoid(upscaled_masks.squeeze(1))

    gt_mask = gt_mask.squeeze(1)
    loss = loss_fn(predicted_masks, gt_mask)
    if train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()
