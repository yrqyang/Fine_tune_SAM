import cv2
import torch
from pathlib import Path
from utils import rgb_to_binary_mask
from segment_anything.utils.transforms import ResizeLongestSide

class DataLoader:
    def __init__(self, base_path, device, sam_model):
        self.base_path = Path(base_path)
        self.device = device
        self.sam_model = sam_model
        self.resize_transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)

    def load_data(self):
        data_paths = self._find_seq_dirs(self.base_path)
        return self._load_images_labels(data_paths)

    def _find_seq_dirs(self, path):
        return {seq_dir.name: {'images': seq_dir / 'Images', 'labels': seq_dir / 'Labels'}
                for seq_dir in path.iterdir() if seq_dir.is_dir() and seq_dir.name.startswith('seq')}

    def _load_images_labels(self, data_paths):
        transformed_data = {}
        for seq, paths in data_paths.items():
            image_paths = sorted(paths['images'].iterdir())
            label_paths = sorted(paths['labels'].iterdir())
            for image_path, label_path in zip(image_paths, label_paths):
                if image_path.stem == label_path.stem:
                    image, label = self._process_image_label(image_path, label_path)
                    transformed_data[image_path.stem] = {
                        'image': image, 'label': label,
                        'original_image_size': image.shape[:2],
                        'input_size': tuple(image.shape[-2:])
                    }
        return transformed_data

    def _process_image_label(self, image_path, label_path):
        # Load and preprocess the image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image = self.resize_transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        transformed_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        # Apply SAM model-specific preprocessing
        processed_image = self.sam_model.preprocess(transformed_image_torch)

        # Process the label
        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        label = rgb_to_binary_mask(label)
        if len(label.shape) == 2:
            label = torch.from_numpy(label).unsqueeze(0).float().to(self.device)
        transformed_label = label.unsqueeze(0)

        return processed_image, transformed_label
