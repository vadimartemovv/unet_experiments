import os
import torch
import logging
import numpy as np
from torch.utils.data import Dataset

class CustomSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Sanitize directories
        self.images = self.sanitize_filenames(os.listdir(image_dir))
        self.masks = self.sanitize_filenames(os.listdir(mask_dir))

        # Ensure the number of images and masks match
        assert len(self.images) == len(self.masks), "Number of images and masks should be the same"

    def sanitize_filenames(self, filenames):
        # Filter out any unwanted files
        valid_extensions = {'.npy'}
        return sorted([f for f in filenames if os.path.splitext(f)[1] in valid_extensions])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        mask_name = os.path.join(self.mask_dir, self.masks[idx])
        
        try:
            image = torch.tensor(np.load(img_name, allow_pickle=True)).float().permute(2, 0, 1)
            mask = torch.tensor(np.load(mask_name, allow_pickle=True)).float()
        except Exception as e:
            logging.error(f"Error loading files at index {idx}: {e}")
            logging.error(f"Image path: {img_name}")
            logging.error(f"Mask path: {mask_name}")
            raise e

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask