"""
Dataset classes for diffusion inpainting with black=mask convention and split mask directories.
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class InpaintingDataset(Dataset):
    """
    Dataset for image inpainting with masks using split-specific mask directories.
    
    Args:
        data_dir (str): Directory containing input images
        mask_dir (str): Directory containing mask subdirectories (train/val/test)
        split (str): Data split ('train', 'val', or 'test') 
        transform (callable, optional): Optional transform to be applied to images
        img_size (int): Size to resize images to
        use_serial_masks (bool): Whether to use masks in serial order (default: True)
        seed (int): Random seed for reproducible mask selection (only used if use_serial_masks=False)
    """
    
    def __init__(self, data_dir, mask_dir, split='train', transform=None, img_size=256, 
                 use_serial_masks=True, seed=42):
        self.data_dir = Path(data_dir)
        self.mask_dir = Path(mask_dir)
        self.split = split
        self.img_size = img_size
        self.use_serial_masks = use_serial_masks
        
        # Default transform
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # RGB to [-1, 1]
        ])

        # Set random seed for reproducible mask selection (if not using serial)
        if not use_serial_masks:
            np.random.seed(seed)

        # Get all image files
        image_extensions = ['.jpg', '.png', '.jpeg', '.bmp', '.tiff']
        self.images = []
        for ext in image_extensions:
            self.images.extend(list(self.data_dir.glob(f'*{ext}')))
            self.images.extend(list(self.data_dir.glob(f'*{ext.upper()}')))
        
        self.images = sorted(list(set(self.images)))  # Remove duplicates and sort
        
        # Get mask files from the appropriate split subdirectory
        mask_split_dir = self.mask_dir / split
        if not mask_split_dir.exists():
            raise ValueError(f"Mask split directory not found: {mask_split_dir}")
        
        self.available_masks = []
        for ext in image_extensions:
            self.available_masks.extend(list(mask_split_dir.glob(f'*{ext}')))
            self.available_masks.extend(list(mask_split_dir.glob(f'*{ext.upper()}')))
        
        self.available_masks = sorted(list(set(self.available_masks)))  # Remove duplicates and sort

        if not self.available_masks:
            raise ValueError(f"No masks found in {mask_split_dir}")
        
        if not self.images:
            raise ValueError(f"No images found in {data_dir}")

        print(f"Found {len(self.images)} images and {len(self.available_masks)} masks for {split} split")
        
        # Create serial mask sequence if using serial ordering
        if self.use_serial_masks:
            self._create_mask_sequence()
            print(f"Using serial mask ordering with {len(self.mask_sequence)} total assignments")
        else:
            print(f"Using random mask selection")

    def _create_mask_sequence(self):
        """Create serial mask sequence that repeats masks as needed."""
        num_images = len(self.images)
        num_masks = len(self.available_masks)
        
        # Calculate how many complete sequences we need
        complete_sequences = (num_images + num_masks - 1) // num_masks
        
        print(f"Creating {complete_sequences} complete sequence(s) of {num_masks} masks for {self.split}")
        
        # Create the full mask sequence (repeat masks as needed)
        self.mask_sequence = []
        for seq_idx in range(complete_sequences):
            for mask_path in self.available_masks:
                self.mask_sequence.append(mask_path)
                if len(self.mask_sequence) >= num_images:
                    break
            if len(self.mask_sequence) >= num_images:
                break
        
        # Trim to exact dataset size
        self.mask_sequence = self.mask_sequence[:num_images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Select mask (serial or random)
        if self.use_serial_masks:
            mask_path = self.mask_sequence[idx]
        else:
            # Random mask selection
            mask_idx = np.random.randint(0, len(self.available_masks))
            mask_path = self.available_masks[mask_idx]
        
        # Load mask
        mask = Image.open(mask_path).convert('L')  # Grayscale mask

        # Apply transforms to image
        image = self.transform(image)
        
        # Transform mask to match image size
        mask_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])
        mask = mask_transform(mask)
        
        # CONVENTION: Black = masked (inpaint), White = retained
        # Invert the mask so that black regions (0) become 1 (inpaint)
        # and white regions (1) become 0 (keep)
        mask = (mask < 0.5).float()  # Black pixels become 1, white pixels become 0
        
        # Create masked image (remove the black regions which are now 1s in mask)
        masked_image = image * (1 - mask)

        return {
            'image': image,               # Original clean image [-1, 1]
            'masked_image': masked_image, # Image with holes (black regions removed)
            'mask': mask,                 # Binary mask (1 = inpaint/was black, 0 = keep/was white)
            'image_path': str(img_path),
            'mask_path': str(mask_path)
        }


class FlatImageDataset(Dataset):
    """Dataset for loading images from a flat directory (no subfolders required)."""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Find all image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
        self.image_paths = []
        
        for ext in image_extensions:
            self.image_paths.extend(list(self.root_dir.glob(f'*{ext}')))
            self.image_paths.extend(list(self.root_dir.glob(f'*{ext.upper()}')))
        
        self.image_paths = sorted(list(set(self.image_paths)))  # Remove duplicates and sort
        
        if not self.image_paths:
            raise ValueError(f"No images found in {root_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'image_path': str(img_path)
        }


class OrderedMaskDataset(Dataset):
    """
    Custom dataset wrapper that cycles through masks in order.
    
    This is useful for inference/testing where you want to apply different masks
    to images in a predictable, repeatable order rather than randomly.
    
    Args:
        base_dataset: Underlying dataset that provides images
        mask_dir (str): Directory containing mask subdirectories (train/val/test)
        split (str): Which mask split to use ('train', 'val', 'test')
        img_size (int): Size to resize images and masks to
    """
    
    def __init__(self, base_dataset, mask_dir, split='test', img_size=256):
        self.base_dataset = base_dataset
        self.mask_dir = Path(mask_dir)
        self.split = split
        self.img_size = img_size
        
        # Get masks from the appropriate split subdirectory
        mask_split_dir = self.mask_dir / split
        if not mask_split_dir.exists():
            raise ValueError(f"Mask split directory not found: {mask_split_dir}")
        
        # Load all mask paths and sort them
        mask_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        self.mask_paths = []
        
        for ext in mask_extensions:
            self.mask_paths.extend(list(mask_split_dir.glob(f'*{ext}')))
            self.mask_paths.extend(list(mask_split_dir.glob(f'*{ext.upper()}')))
        
        self.mask_paths = sorted(list(set(self.mask_paths)))  # Remove duplicates and sort
        
        if not self.mask_paths:
            raise ValueError(f"No masks found in {mask_split_dir}")
        
        print(f"Found {len(self.mask_paths)} masks in {mask_split_dir}")
        print(f"First few masks: {[p.name for p in self.mask_paths[:5]]}")
        
        # Transforms for masks
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        
        # Transforms for images (in case base dataset doesn't have them)
        self.image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
        ])
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get base image
        base_item = self.base_dataset[idx]
        
        # Handle different dataset types
        if isinstance(base_item, dict):
            # FlatImageDataset or similar returns dict
            image = base_item['image']
            image_path = base_item.get('image_path', None)
        elif isinstance(base_item, tuple):
            # ImageFolder returns tuple (image, class)
            image = base_item[0]
            image_path = getattr(self.base_dataset, 'imgs', [None])[idx]
            if image_path:
                image_path = image_path[0] if isinstance(image_path, tuple) else image_path
        else:
            # Direct tensor
            image = base_item
            image_path = None
        
        # Apply image transform if needed (e.g., if base dataset doesn't normalize)
        if not isinstance(image, torch.Tensor):
            image = self.image_transform(image)
        
        # Select mask in order (cycle through if we run out)
        mask_idx = idx % len(self.mask_paths)
        mask_path = self.mask_paths[mask_idx]
        
        # Load and process mask
        mask_pil = Image.open(mask_path).convert('L')  # Convert to grayscale
        mask = self.mask_transform(mask_pil)
        
        # CONVENTION: Black = masked (inpaint), White = retained
        # Invert the mask so that black regions (0) become 1 (inpaint)
        # and white regions (1) become 0 (keep)
        mask = (mask < 0.5).float()  # Black pixels become 1, white pixels become 0
        
        # Create masked image (remove the black regions which are now 1s in mask)
        masked_image = image * (1 - mask)
        
        return {
            'image': image,
            'masked_image': masked_image,
            'mask': mask,
            'image_path': image_path,
            'mask_path': str(mask_path),
            'mask_idx': mask_idx
        }


def create_inpainting_dataloaders(train_dir, val_dir, mask_dir, batch_size=4, 
                                 img_size=256, num_workers=4, pin_memory=True,
                                 use_serial_masks=True):
    """
    Create training and validation dataloaders for inpainting with split-specific masks.
    
    Args:
        train_dir (str): Training images directory
        val_dir (str): Validation images directory  
        mask_dir (str): Masks base directory (should contain train/val/test subdirs)
        batch_size (int): Batch size
        img_size (int): Image size
        num_workers (int): Number of dataloader workers
        pin_memory (bool): Whether to pin memory
        use_serial_masks (bool): Whether to use serial mask ordering
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    
    # Create datasets with split-specific mask directories
    train_dataset = InpaintingDataset(
        train_dir, mask_dir, 
        split='train',  # Use train mask split
        img_size=img_size, 
        use_serial_masks=use_serial_masks,
        seed=42
    )
    
    val_dataset = InpaintingDataset(
        val_dir, mask_dir, 
        split='val',  # Use val mask split
        img_size=img_size, 
        use_serial_masks=use_serial_masks,
        seed=99  # Different seed for validation (only matters if use_serial_masks=False)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader


# Convenience function for creating dataloaders with explicit serial mask ordering
def create_serial_mask_dataloaders(train_dir, val_dir, mask_dir, batch_size=4, 
                                  img_size=256, num_workers=4, pin_memory=True):
    """
    Create dataloaders with guaranteed serial mask ordering.
    This is a convenience wrapper around create_inpainting_dataloaders.
    """
    return create_inpainting_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir, 
        mask_dir=mask_dir,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        use_serial_masks=True
    )


def create_inference_dataloader(test_dir, mask_dir, split='test', batch_size=4, img_size=256, 
                               num_workers=4, pin_memory=True, random_samples=0):
    """
    Create a dataloader for inference/testing with ordered mask cycling.
    
    Args:
        test_dir (str): Directory containing test images
        mask_dir (str): Directory containing mask subdirectories (train/val/test)
        split (str): Which mask split to use ('test', 'train', 'val')
        batch_size (int): Batch size
        img_size (int): Image size
        num_workers (int): Number of dataloader workers
        pin_memory (bool): Whether to pin memory
        random_samples (int): Number of random samples to select (0 = use all)
        
    Returns:
        DataLoader: Configured dataloader for inference
    """
    from torch.utils.data import Subset
    import random
    
    # Create image transform
    image_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
    ])
    
    # Try ImageFolder first, then fallback to FlatImageDataset
    try:
        from torchvision.datasets import ImageFolder
        base_dataset = ImageFolder(test_dir, transform=image_transform)
        print(f"Found {len(base_dataset)} images in {test_dir} (ImageFolder)")
    except (FileNotFoundError, RuntimeError):
        # Fallback to custom flat directory dataset
        print(f"ImageFolder failed, using flat directory loader for {test_dir}")
        base_dataset = FlatImageDataset(test_dir, transform=image_transform)
        print(f"Found {len(base_dataset)} images in {test_dir} (FlatImageDataset)")
    
    # Wrap with ordered mask dataset using the specified split
    full_dataset = OrderedMaskDataset(
        base_dataset=base_dataset,
        mask_dir=mask_dir,
        split=split,  # Use the specified mask split
        img_size=img_size
    )
    
    # Apply random sampling if requested
    if random_samples > 0:
        print(f"Selecting {random_samples} random samples from {len(full_dataset)} total samples")
        
        # Get random indices
        all_indices = list(range(len(full_dataset)))
        random.shuffle(all_indices)
        selected_indices = all_indices[:random_samples]
        
        # Create subset
        dataset = Subset(full_dataset, selected_indices)
        print(f"Using {len(dataset)} randomly selected samples")
    else:
        dataset = full_dataset
        print(f"Using all {len(dataset)} samples")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle to maintain mask order
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    print(f"Total batches: {len(dataloader)}")
    
    return dataloader
