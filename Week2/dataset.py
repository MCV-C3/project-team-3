import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

class C3Dataset(ImageFolder):
    def __init__(self, root, transform=None, device=None):
        super().__init__(root, transform=transform)
        
        print(f"Loading {len(self.samples)} images into memory...")
        self.images = []
        self.targets = []
        
        # We iterate over self.samples (provided by ImageFolder), which contains (path, class_index)
        for path, class_index in tqdm(self.samples):
            loaded_image = pil_to_tensor(self.loader(path))
            self.images.append(loaded_image)
            self.targets.append(class_index)
        
        self.images = torch.stack(self.images).to(device=device)
        self.targets = torch.tensor(self.targets, device=device)
    
    def len(self):
        return len(self.images)

    def __getitem__(self, index):
        sample = self.images[index]
        target = self.targets[index]
        
        if self.transform is not None:
            sample = self.transform(sample)
            
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
