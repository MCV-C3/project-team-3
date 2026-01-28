import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

class SceneDataset(ImageFolder):
    def __init__(self, root, transform=None, device=None):
        super().__init__(root, transform=transform)
        
        print(f"Loading {len(self.samples)} images into memory...")
        self.images = []
        self.labels = []
        
        # We iterate over self.samples (provided by ImageFolder), which contains (path, class_index)
        for path, class_index in tqdm(self.samples):
            loaded_image = pil_to_tensor(self.loader(path))
            self.images.append(loaded_image)
            self.labels.append(class_index)
        
        self.images = torch.stack(self.images).to(dtype=torch.float32, device=device) / 255
        self.labels = torch.tensor(self.labels, device=device)
    
    def len(self):
        return len(self.images)

    def __getitem__(self, index):
        sample = self.images[index]
        target = self.labels[index]
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target


def get_transforms(augment=True, mean=None, std=None):
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
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            # transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        # Validation/test transforms without augmentation
        transform = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    
    return transform

def create_data_loaders(train_path, val_path, device,
                        batch_size=32, num_workers=4,
                        mean=None, std=None):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        train_paths: Training data
        val_paths: Validation data
        device: Training device
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        mean, std: Normalization parameters
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Get transforms
    train_transform = get_transforms(augment=True, mean=mean, std=std)
    val_transform = get_transforms(augment=False, mean=mean, std=std)
    
    # Create datasets
    train_dataset = SceneDataset(train_path, transform=train_transform, device=device)
    val_dataset = SceneDataset(val_path, transform=val_transform, device=device)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader