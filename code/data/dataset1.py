"""
Dataset classes for diffusion inpainting.
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader


def get_transform(img_size=256):
    """Get image transform without torchvision."""
    def transform(image):
        # Resize
        image = image.resize((img_size, img_size))
        # Convert to tensor
        tensor = torch.from_numpy(np.array(image)).float() / 255.0
        if tensor.dim() == 2:  # Grayscale
            tensor = tensor.unsqueeze(0)
        else:  # RGB
            tensor = tensor.permute(2, 0, 1)
        # Normalize to [-1, 1]
        tensor = tensor * 2.0 - 1.0
        return tensor
    return transform


def get_mask_transform(img_size=256):
    """Get mask transform without torchvision."""
    def transform(mask):
        # Resize
        mask = mask.resize((img_size, img_size))
        # Convert to tensor
        tensor = torch.from_numpy(np.array(mask)).float() / 255.0
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        return tensor
    return transform


class InpaintingDataset(Dataset):
    """
    Dataset for image inpainting with masks.
    
    Args:
        data_dir (str): Directory containing input images
        mask_dir (str): Directory containing mask images
        img_size (int): Size to resize images to
        seed (int): Random seed for reproducible mask selection
    """
    
    def __init__(self, data_dir, mask_dir, img_size=256, seed=42):
        self.data_dir = Path(data_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        
        # Set random seed
        np.random.seed(seed)
        
        # Get transforms
        self.img_transform = get_transform(img_size)
        self.mask_transform = get_mask_transform(img_size)

        # Get all image files
        self.images = sorted([
            f for f in os.listdir(data_dir) 
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff'))
        ])
        
        # Get all mask files
        self.available_masks = sorted([
            f for f in os.listdir(mask_dir) 
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff'))
        ])

        if not self.available_masks:
            raise ValueError(f"No masks found in {mask_dir}")
        
        if not self.images:
            raise ValueError(f"No images found in {data_dir}")

        print(f"Found {len(self.images)} images and {len(self.available_masks)} masks")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = self.data_dir / self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Randomly select a mask
        mask_idx = np.random.randint(0, len(self.available_masks))
        mask_path = self.mask_dir / self.available_masks[mask_idx]
        mask = Image.open(mask_path).convert('L')  # Grayscale mask

        # Apply transforms
        image = self.img_transform(image)
        mask = self.mask_transform(mask)
        
        # Create masked image (image with holes)
        masked_image = image * (1 - mask)

        return {
            'image': image,           # Original clean image
            'masked_image': masked_image,  # Image with holes
            'mask': mask,            # Binary mask (1 = hole, 0 = keep)
            'image_path': str(img_path),
            'mask_path': str(mask_path)
        }


def create_inpainting_dataloaders(train_dir, val_dir, mask_dir, batch_size=4, 
                                 img_size=256, num_workers=4, pin_memory=True):
    """
    Create training and validation dataloaders for inpainting.
    
    Args:
        train_dir (str): Training images directory
        val_dir (str): Validation images directory  
        mask_dir (str): Masks directory
        batch_size (int): Batch size
        img_size (int): Image size
        num_workers (int): Number of dataloader workers
        pin_memory (bool): Whether to pin memory
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    
    # Create datasets
    train_dataset = InpaintingDataset(
        train_dir, mask_dir, 
        img_size=img_size, 
        seed=42
    )
    
    val_dataset = InpaintingDataset(
        val_dir, mask_dir, 
        img_size=img_size, 
        seed=99  # Different seed for validation
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader
