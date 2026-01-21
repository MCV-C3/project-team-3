"""
Dataset class for scene classification.
"""

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split


class SceneDataset(Dataset):
    """
    Custom dataset for scene classification.
    """
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths (list): List of paths to images
            labels (list): List of labels (integers)
            transform (callable, optional): Optional transform to apply
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(image_size=128, augment=True, mean=None, std=None):
    """
    Get data transforms for training and validation.
    
    Args:
        image_size (int): Target image size
        augment (bool): Whether to apply data augmentation
        mean (list): Normalization mean
        std (list): Normalization std
        
    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    if augment:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        # Validation/test transforms without augmentation
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    
    return transform


def load_dataset_from_folders(data_dir, categories, train_split=0.7, val_split=0.15, seed=42):
    """
    Load dataset from folder structure where each category is a subfolder.
    
    Expected structure:
        data_dir/
            category1/
                img1.jpg
                img2.jpg
            category2/
                img1.jpg
                img2.jpg
    
    Args:
        data_dir (str or Path): Root directory containing category folders
        categories (list): List of category names
        train_split (float): Proportion for training set
        val_split (float): Proportion for validation set
        seed (int): Random seed for splitting
        
    Returns:
        tuple: (train_paths, train_labels, val_paths, val_labels, test_paths, test_labels)
    """
    data_dir = Path(data_dir)
    all_image_paths = []
    all_labels = []
    
    # Collect all images and labels
    for label_idx, category in enumerate(categories):
        category_dir = data_dir / category
        if not category_dir.exists():
            print(f"Warning: Category directory {category_dir} does not exist")
            continue
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        for ext in image_extensions:
            image_files = list(category_dir.glob(f'*{ext}'))
            all_image_paths.extend(image_files)
            all_labels.extend([label_idx] * len(image_files))
    
    print(f"Total images found: {len(all_image_paths)}")
    
    # Convert to numpy arrays for splitting
    all_image_paths = np.array(all_image_paths)
    all_labels = np.array(all_labels)
    
    # First split: separate test set
    test_split = 1.0 - train_split - val_split
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        all_image_paths, all_labels, 
        test_size=test_split, 
        random_state=seed, 
        stratify=all_labels
    )
    
    # Second split: separate train and validation
    val_ratio = val_split / (train_split + val_split)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=val_ratio,
        random_state=seed,
        stratify=train_val_labels
    )
    
    print(f"Train set: {len(train_paths)} images")
    print(f"Validation set: {len(val_paths)} images")
    print(f"Test set: {len(test_paths)} images")
    
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def load_presplit_dataset(data_dir, categories, val_split=0.15, seed=42):
    """
    Load dataset that already has train/test splits.
    
    Expected structure:
        data_dir/
            train/
                category1/
                    img1.jpg
                category2/
                    img1.jpg
            test/
                category1/
                    img1.jpg
                category2/
                    img1.jpg
    
    Args:
        data_dir (str or Path): Root directory containing train/test folders
        categories (list): List of category names
        val_split (float): Proportion of training set to use for validation
        seed (int): Random seed for splitting
        
    Returns:
        tuple: (train_paths, train_labels, val_paths, val_labels, test_paths, test_labels)
    """
    data_dir = Path(data_dir)
    
    # Load training data
    train_dir = data_dir / "train"
    train_paths_all = []
    train_labels_all = []
    
    for label_idx, category in enumerate(categories):
        category_dir = train_dir / category
        if not category_dir.exists():
            print(f"Warning: Category directory {category_dir} does not exist")
            continue
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        for ext in image_extensions:
            image_files = list(category_dir.glob(f'*{ext}'))
            train_paths_all.extend(image_files)
            train_labels_all.extend([label_idx] * len(image_files))
    
    print(f"Total training images found: {len(train_paths_all)}")
    
    # Load test data
    test_dir = data_dir / "test"
    test_paths = []
    test_labels = []
    
    for label_idx, category in enumerate(categories):
        category_dir = test_dir / category
        if not category_dir.exists():
            print(f"Warning: Category directory {category_dir} does not exist")
            continue
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        for ext in image_extensions:
            image_files = list(category_dir.glob(f'*{ext}'))
            test_paths.extend(image_files)
            test_labels.extend([label_idx] * len(image_files))
    
    print(f"Total test images found: {len(test_paths)}")
    
    # Convert to numpy arrays
    train_paths_all = np.array(train_paths_all)
    train_labels_all = np.array(train_labels_all)
    test_paths = np.array(test_paths)
    test_labels = np.array(test_labels)
    
    # Split training data into train and validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths_all, train_labels_all,
        test_size=val_split,
        random_state=seed,
        stratify=train_labels_all
    )
    
    print(f"Train set: {len(train_paths)} images")
    print(f"Validation set: {len(val_paths)} images")
    print(f"Test set: {len(test_paths)} images")
    
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels



def create_data_loaders(train_paths, train_labels, val_paths, val_labels, 
                        test_paths, test_labels, batch_size=32, num_workers=4,
                        image_size=128, mean=None, std=None):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        train_paths, train_labels: Training data
        val_paths, val_labels: Validation data
        test_paths, test_labels: Test data
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        image_size (int): Image size
        mean, std: Normalization parameters
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Get transforms
    train_transform = get_transforms(image_size, augment=True, mean=mean, std=std)
    val_transform = get_transforms(image_size, augment=False, mean=mean, std=std)
    
    # Create datasets
    train_dataset = SceneDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = SceneDataset(val_paths, val_labels, transform=val_transform)
    test_dataset = SceneDataset(test_paths, test_labels, transform=val_transform)
    
    # Create data loaders
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
