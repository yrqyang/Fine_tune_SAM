import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2
import sys
import os
from segment_anything import sam_model_registry, SamPredictor
from skimage import measure
from tqdm import tqdm

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

def show_mask(
    mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def create_vehicle_mask(bounding_boxes, image_shape):
    # Create a blank binary mask with the same dimensions as the image
    vehicle_mask = np.zeros(image_shape[:2], dtype=np.uint8)

    # Iterate through each bounding box
    for box in bounding_boxes:
        # Extract the corners
        pts = np.array([[box[0], box[1]],
                        [box[2], box[3]],
                        [box[4], box[5]],
                        [box[6], box[7]]], np.int32)

        # Reshape pts to a 3D array for cv2.fillPoly
        pts = pts.reshape((-1, 1, 2))

        # Draw the polygon on the mask
        cv2.fillPoly(vehicle_mask, [pts], 1)

    return vehicle_mask

def interpolate_frames(frames, target_frame_count):
    input_frame_count = len(frames)
    interpolated_frames = []
    for i in range(target_frame_count):
        alpha = i * (input_frame_count - 1) / (target_frame_count - 1)
        left_frame_index = int(np.floor(alpha))
        right_frame_index = min(left_frame_index + 1, input_frame_count - 1)
        blend_alpha = alpha - left_frame_index
        interpolated_frame = cv2.addWeighted(frames[left_frame_index], 1 - blend_alpha, frames[right_frame_index], blend_alpha, 0)
        interpolated_frames.append(interpolated_frame)
    return interpolated_frames

def create_video_from_images(image_folder, output_video_path, target_frame_rate=30):
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".jpg")], key=lambda x: int(x.split('_')[1].split('.')[0]))

    frames = []
    for filename in tqdm(frame_files):
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error reading image: {image_path}")
            continue

        frames.append(image)
    
    # Interpolate frames to increase frame rate
    original_frame_rate = 5  # 25 (original fps) /5 (sample frequency) = 5
    total_frames_after_interpolation = len(frames) * (target_frame_rate // original_frame_rate)
    smooth_frames = interpolate_frames(frames, total_frames_after_interpolation)

    output_video = cv2.VideoWriter(output_video_path, fourcc, target_frame_rate)

    for frame in tqdm(smooth_frames):
        output_video.write(frame)

    # Release the video writer
    output_video.release()

    print("Video processing complete. Output saved to:", output_video_path)

def main():
    sys.path.append("..")
    sam_ft_ckpt = "../models/sam_model_h.pth"
    model_type = "vit_h"

    device = "cuda"

    sam_ft = sam_model_registry[model_type](checkpoint=sam_ft_ckpt)
    sam_ft.to(device=device)

    predictor = SamPredictor(sam_ft)

    # Directory containing images and output video path
    output_video_path = "../Road_Segmentation_Galatsi.avi"
    Image_PATH = "../dataset/Galatsi/"
    VehicleMask_PATH = "../Vehicle_Detection/"
    TEMP_SAVING_PATH = "../temp/"

    # Process each image in the dataset directory and save to temp directory 
    for filename in tqdm(sorted(os.listdir(Image_PATH))):
        if filename.endswith(".jpg"):
            image_path = os.path.join(Image_PATH, filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
            # Perform segmentation
            image_cv = np.array(image)
            predictor.set_image(image_cv)
            sam_masks, _, _ = predictor.predict(
                point_coords=None, 
                point_labels=None, 
                multimask_output=False,
                )
            
            # Add vehicle mask
            # Path to the results file
            vehicle_mask_name = os.path.splitext(filename)[0]
            vehicle_mask_path = os.path.join(VehicleMask_PATH, f"{vehicle_mask_name}.txt")

            # Initialize an empty list to store all the coordinates
            coordinates_list = []

            # Read the file
            with open(vehicle_mask_path, 'r') as file:
                for line in file:
                    # Split the line into components
                    components = line.strip().split(',')
                    # Extract the coordinate points and convert them to float
                    coords = list(map(float, components[:8]))
                    # Append the coordinates to the list
                    coordinates_list.append(coords)

            # Convert the list of coordinates into a PyTorch tensor
            boxes = torch.tensor(coordinates_list)

            vehicle_mask = create_vehicle_mask(boxes, np.array(image).shape[0:2])

            seg_mask = sam_masks + vehicle_mask # Combine extracted vehicle mask with the road mask
            seg_mask[seg_mask == 2] = 1 # the overlapping pixels of two masks

            # Post-process the segmentation mask
            sam_connect_mask = seg_mask.astype(np.uint8)
            kernel = np.ones((5, 5), np.uint8)  # kernel for morphological operations
            # Perform morphological closing to connect regions
            closed_mask = cv2.morphologyEx(sam_connect_mask.squeeze(), cv2.MORPH_CLOSE, kernel, iterations=16)
        
            labeled_mask, num_labels = measure.label(closed_mask, return_num=True, background=0)
            label_props = measure.regionprops(labeled_mask)
            area_threshold = 1000
            filtered_mask = np.zeros_like(labeled_mask)
            for prop in label_props:
                if prop.area > area_threshold:
                    filtered_mask[labeled_mask == prop.label] = 1
            sam_connect_mask = filtered_mask[np.newaxis, ...]

            # Mask on image
            height, width = image.shape[:2]
            figsize = width / 100, height / 100  # 100 DPI, to preserve the resolution
    
            plt.figure(figsize=figsize, dpi=100)
            # plt.figure(figsize=(10, 10))
            plt.imshow(image)
            show_mask(sam_connect_mask, plt.gca())
            plt.axis('off')

            # Save the segmented image to a file and add to video
            seg_path = os.path.join(TEMP_SAVING_PATH, f"{filename}")
            plt.savefig(seg_path, bbox_inches='tight', pad_inches=0, dpi=100) # same as the setting above
            plt.close()

    create_video_from_images(TEMP_SAVING_PATH, output_video_path)


if __name__ == '__main__':
    main()