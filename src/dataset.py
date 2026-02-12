"""
Dataset module for loading and preprocessing chest X-ray images.
Handles both training and inference pipelines.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Callable

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class ChestXRayDataset(Dataset):
    """
    Custom Dataset for chest X-ray images.
    
    Args:
        root_dir (str): Path to the dataset directory
        split (str): 'train', 'val', or 'test'
        transform (callable, optional): Transforms to apply to images
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.classes = ['NORMAL', 'PNEUMONIA']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Collect all image paths
        self.images = []
        self.labels = []
        
        split_dir = self.root_dir / split
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} does not exist")
        
        for class_name in self.classes:
            class_dir = split_dir / class_name
            if class_dir.exists():
                for img_file in class_dir.glob('*.jpeg'):
                    self.images.append(str(img_file))
                    self.labels.append(self.class_to_idx[class_name])
                # Also check for .jpg extension
                for img_file in class_dir.glob('*.jpg'):
                    self.images.append(str(img_file))
                    self.labels.append(self.class_to_idx[class_name])
                # Check for .png extension
                for img_file in class_dir.glob('*.png'):
                    self.images.append(str(img_file))
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get image and label by index.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            Tuple[torch.Tensor, int]: Image tensor and label
        """
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image in grayscale
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(image_size: int = 224, augment: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get preprocessing and augmentation transforms.
    
    Args:
        image_size (int): Target image size (default: 224 for ResNet)
        augment (bool): Whether to apply augmentation (for training)
        
    Returns:
        Tuple[transforms.Compose, transforms.Compose]: Train and validation transforms
    """
    # ImageNet normalization statistics
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Validation/Test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    
    return train_transform, val_transform


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for train, validation, and test sets.
    
    Args:
        data_dir (str): Path to the dataset directory
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of workers for data loading
        image_size (int): Target image size
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, val, and test dataloaders
    """
    train_transform, val_transform = get_transforms(image_size=image_size, augment=True)
    
    # Create datasets
    train_dataset = ChestXRayDataset(data_dir, split='train', transform=train_transform)
    val_dataset = ChestXRayDataset(data_dir, split='val', transform=val_transform)
    test_dataset = ChestXRayDataset(data_dir, split='test', transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
