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

def create_video_from_images(image_folder, output_video_path):
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_size = (1280, 720)  # Specify the unified frame size
    output_video = cv2.VideoWriter(output_video_path, fourcc, 10.0, frame_size)

    # Process each image in the temporary folder
    for filename in tqdm(sorted(os.listdir(image_folder), key=lambda x: int(x.split('_')[1].split('.')[0]))):
        if filename.endswith(".jpg"): 
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Error reading image: {image_path}")
                continue

            if image.shape[1] != 1280 or image.shape[0] != 720:
                image = cv2.resize(image, frame_size)

            output_video.write(image)

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

            # Post-process the segmentation mask
            sam_connect_mask = seg_mask.astype(np.uint8)
            kernel = np.ones((7, 7), np.uint8)
            eroded_mask = cv2.erode(sam_connect_mask.squeeze(), kernel, iterations=1)
            dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=10)
            closed_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, kernel, iterations=6)
        
            labeled_mask, num_labels = measure.label(closed_mask, return_num=True, background=0)
            label_props = measure.regionprops(labeled_mask)
            area_threshold = 1000
            filtered_mask = np.zeros_like(labeled_mask)
            for prop in label_props:
                if prop.area > area_threshold:
                    filtered_mask[labeled_mask == prop.label] = 1
            sam_connect_mask = filtered_mask[np.newaxis, ...]

            # Mask on image
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            show_mask(sam_connect_mask, plt.gca())
            plt.axis('off')

            # Save the segmented image to a file and add to video
            seg_path = os.path.join(TEMP_SAVING_PATH, f"{filename}")
            plt.savefig(seg_path, bbox_inches='tight', pad_inches=0)
            plt.close()

    create_video_from_images(TEMP_SAVING_PATH, output_video_path)


if __name__ == '__main__':
    main()