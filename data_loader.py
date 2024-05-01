import numpy as np
import cv2
import torch
from utils import rgb_to_binary_mask
from segment_anything.utils.transforms import ResizeLongestSide
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random

class DataLoader:
    def __init__(self, base_paths, device, sam_model):
        self.base_paths = base_paths
        self.device = device
        self.sam_model = sam_model
        self.resize_transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)
        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ])

    def load_data(self):
        transformed_data = {}
        for base_path in self.base_paths:
            data_paths = self._find_seq_dirs(base_path['path'], is_new=base_path['new_dataset'])
            transformed_data.update(self._load_images_labels(data_paths, base_path['new_dataset']))
        return transformed_data

    def _find_seq_dirs(self, path, is_new=False):
        if not is_new:
            # Old dataset format
            return {seq_dir.name: {'images': seq_dir / 'Images', 'labels': seq_dir / 'Labels'}
                    for seq_dir in path.iterdir() if seq_dir.is_dir() and seq_dir.name.startswith('seq')}
        else:
            # New dataset format
            return {'epflml': {'images': path / 'images', 'labels': path / 'groundtruth'}}

    def _load_images_labels(self, data_paths, is_new_dataset):
        transformed_data = {}
        for seq, paths in data_paths.items():
            image_paths = sorted(paths['images'].iterdir())
            label_paths = sorted(paths['labels'].iterdir())
            for image_path, label_path in zip(image_paths, label_paths):
                if image_path.stem == label_path.stem:
                    image, mask, input_size, original_image_size = self._process_image_label(image_path, label_path, is_new_dataset)
                    transformed_data[seq + '/' + image_path.stem] = {
                        'image': image, 'mask': mask,
                        'original_image_size': original_image_size,
                        'input_size': input_size
                    }
        return transformed_data

    def _process_image_label(self, image_path, label_path, is_new_dataset):
        # Load and preprocess the image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)

        # Data Augmentation
        if is_new_dataset:
            # Convert to PIL for augmentation
            image_pil = TF.to_pil_image(image)
            label_pil = TF.to_pil_image(label)

            # Synchronize transforms
            seed = random.randint(0, 2**32)
            random.seed(seed)
            torch.manual_seed(seed)
            image_pil = self.augmentations(image_pil)
            random.seed(seed)
            torch.manual_seed(seed)
            label_pil = self.augmentations(label_pil)

            image = np.array(image_pil)
            label = np.array(label_pil)
        else:
            label = rgb_to_binary_mask(label)

        input_image = self.resize_transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        transformed_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        # Apply SAM model-specific preprocessing
        processed_image = self.sam_model.preprocess(transformed_image_torch)
        original_image_size = image.shape[:2]
        input_size = tuple(transformed_image_torch.shape[-2:])

        # Process the label
        if len(label.shape) == 2: # Just H and W dimensions
            label = torch.from_numpy(label).unsqueeze(0) # Convert to tensor and add channel dimension -> [1, H, W]
        label = label.to(self.device).float()
        transformed_label = label.unsqueeze(0) # Add batch dimension -> [1, 1, H, W] 

        return processed_image, transformed_label, input_size, original_image_size
