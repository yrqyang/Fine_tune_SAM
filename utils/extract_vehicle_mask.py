######## Change the image path, results saving path, and the configuration path accordingly #########

from geopy.distance import distance
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from tqdm import tqdm

import mmcv
from mmcv import collect_env, Config
from mmcv.ops import nms_rotated
from mmcv.runner import load_checkpoint

from mmdet.apis import inference_detector, set_random_seed, show_result_pyplot, train_detector
from mmdet.datasets import build_dataset
from mmdet.datasets.custom import CustomDataset
from mmdet.models import build_detector

from mmrotate import evaluation
from mmrotate.apis import inference_detector_by_patches
from mmrotate.core import obb2poly_np, poly2obb_np
from mmrotate.core.evaluation import eval_rbbox_map
from mmrotate.datasets.builder import ROTATED_DATASETS
from mmrotate.datasets.dota import DOTADataset

#from CustomSuperGlue import *
#from CustomSIFT import *

def from_rbbox_to_4pts(bbox):
    """Convert oriented bounding boxes to polygons.
    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    xc, yc, w, h, ag = bbox[:5]
    wx, wy = w / 2 * np.cos(ag), w / 2 * np.sin(ag)
    hx, hy = -h / 2 * np.sin(ag), h / 2 * np.cos(ag)
    p1 = (xc - wx - hx, yc - wy - hy)
    p2 = (xc + wx - hx, yc + wy - hy)
    p3 = (xc + wx + hx, yc + wy + hy)
    p4 = (xc - wx + hx, yc - wy + hy)
    poly = np.int0(np.array([p1, p2, p3, p4]))
        
    return poly, bbox[-1]

def result_filtering_class(result, classes, subclasses, shrink=False):
    if shrink:
        out = [result[classes.index(c)] for c in subclasses]
    else:
        out = [result[classes.index(c)] if c in subclasses else np.array([]).reshape((0, 6)) for c in classes]
    
    return out


def result_filtering_score(result, score):
    filtered = []
    for result_c in result:
        if len(result_c) == 0:
            filtered.append(np.array([]).reshape((0, 6)))
        else:
            tmp = [r for r in result_c if r[-1]>=score]
            if len(tmp)>0:
                filtered.append(np.array(tmp))
            else:
                filtered.append(np.array([]).reshape((0, 6)))
    
    return filtered


def result_filtering(result, score, classes, subclasses):
    result_f1 = result_filtering_class(result, classes, subclasses)
    result_f2 = result_filtering_score(result_f1, score)
    
    return result_f2

def process_image(image_path, output_dir, model):
    img = cv2.imread(image_path)
    result = inference_detector_by_patches(model, img, [1300], [200], [1.0], 0.1)
    result = result_filtering(result, 0.5, model.CLASSES, subclasses=["small-vehicle", "large-vehicle"])

    # Generate output file path
    base_name = os.path.basename(image_path)
    output_file = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}.txt")
    
    with open(output_file, "w") as f:
        for c in ["small-vehicle", "large-vehicle"]:
            for r in result[list(model.CLASSES).index(c)]:
                bb_pts, prob = from_rbbox_to_4pts(r)
                x1, y1, x2, y2, x3, y3, x4, y4 = bb_pts.flatten()
                f.write("{},{},{},{},{},{},{},{},{},{}\n".format(x1, y1, x2, y2, x3, y3, x4, y4, c, prob))

def main():
    # Choose to use a config and initialize the detector
    config = '/home/student/mmrotate/configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py'
    # Setup a checkpoint file to load
    checkpoint = '/home/student/mmrotate/checkpoints/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth'

    # Directory of images
    input_dir = "../../dataset/Galatsi/"
    # Directory to save the results
    output_dir = "../../Vehicle_Detection/"
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set the device to be used for evaluation
    device='cuda:0'

    # Load the config
    config = mmcv.Config.fromfile(config)
    # Set pretrained to be None since we do not need pretrained model here
    config.model.pretrained = None

    # Initialize the detector
    model = build_detector(config.model)

    # Load checkpoint
    checkpoint = load_checkpoint(model, checkpoint, map_location=device)

    # Set the classes of models for inference
    model.CLASSES = checkpoint['meta']['CLASSES']

    # We need to set the model's cfg for inference
    for k in config.keys():
        print("=====================")
        print(k)
        print(config[k])
    model.cfg = config

    # Convert the model to GPU
    model.to(device)
    # Convert the model into evaluation mode
    model.eval()

    # List of all images
    image_files = [f for f in os.listdir(input_dir) if f.endswith(".jpg")] # all images are .jpg

    # Loop through all images in the input directory with a progress bar
    for image_file in tqdm(image_files, desc="Processing Images"):
        image_path = os.path.join(input_dir, image_file)
        process_image(image_path, output_dir, model)

    print("Processing complete.")

if __name__ == '__main__':
    main()