import csv
import json
import os
from typing import *

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as F
import wandb
from dataset import C3Dataset
from models import FlexibleMlp
from sklearn.svm import SVC
from torch.utils.data import DataLoader


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_csv(path: str, rows: List[Dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def make_transforms(resize: int):
    train_transforms = F.Compose([
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        F.Resize(size=(resize, resize))
    ])
    val_transforms = F.Compose([
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        F.Resize(size=(resize, resize))
    ])
    return train_transforms, val_transforms


def make_loaders(train_dir: str, test_dir: str, resize: int, batch_size: int,
                 val_ratio: float = 0.2, num_workers: int = 8, device=None):
    train_transforms, val_transforms = make_transforms(resize)

    train_ds = C3Dataset(train_dir, transform=train_transforms, device=device)
    val_ds = C3Dataset(test_dir, transform=val_transforms, device=device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False)

    sample_img, _ = train_ds[0]
    C, H, W = sample_img.shape
    input_d = C * H * W

    num_classes = len(train_ds.classes)

    return train_loader, val_loader, input_d, num_classes


if __name__ == "__main__":
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    set_seed(42)

    # ---- EDITA ESTO ----
    TRAIN_DIR = "/home/msiau/data/tmp/agarciat/MCVC/C3/places_reduced/train"
    TEST_DIR  = "/home/msiau/data/tmp/agarciat/MCVC/C3/places_reduced/val"
    # --------------------

    WANDB_PROJECT = "MCVCC3-Team3-2526"
    WANDB_ENTITY = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carga best fine
    with open("results/best_coarse_config.json", "r") as f:
        best_coarse = json.load(f)
    
    best_model = torch.load('results/checkpoints/fine_r128_b256_h[128, 128].pt')

    resize = int(best_coarse["resize"])
    batch_size = int(best_coarse["batch_size"])
    hidden_dims = best_model['hidden_dims']

    print("Using best fine:", {'resize': resize, 'batch_size': batch_size, 'hidden_dims': hidden_dims})

    # Loaders fijos
    train_loader, val_loader, input_d, num_classes = make_loaders(
        TRAIN_DIR, TEST_DIR, resize=resize, batch_size=batch_size, val_ratio=0.2, num_workers=0, device=device
    )

    model = FlexibleMlp(input_d=input_d, hidden_dims=hidden_dims, output_d=num_classes)
    model.load_state_dict(best_model['model_state_dict'], strict=False)
    model = model.to(device=device)
    
    train_x = []
    train_y = []
    val_x = []
    val_y = []

    with torch.no_grad():
        model.eval()

        img: torch.Tensor
        labels: torch.Tensor
        for img, labels in train_loader:
            features: torch.Tensor = model(img)
            train_x.append(features.cpu().detach().numpy())
            train_y.append(labels.cpu().detach().numpy())
        train_x = np.concat(train_x, axis=0)
        train_y = np.concat(train_y, axis=0)

        img: torch.Tensor
        labels: torch.Tensor
        for img, labels in train_loader:
            features: torch.Tensor = model(img)
            val_x.append(features.cpu().detach().numpy())
            val_y.append(labels.cpu().detach().numpy())
        val_x = np.concat(val_x, axis=0)
        val_y = np.concat(val_y, axis=0)
    
    svm_kernels = ['linear', 'poly', 'rbf', 'sigmoid']

    results = []
    group_name = f"svm_r{resize}_b{batch_size}_h[128, 128]"

    for kernel in svm_kernels:
        print(f"Training SVM with kernel {kernel}")

        svm = SVC(kernel=kernel)
        svm = svm.fit(train_x, train_y)
        y_pred = svm.predict(val_x)
        acc = np.sum(y_pred == val_y) / len(val_y)
        print(f"SVM accuracy: {acc}")
        result = {
            'resize': resize,
            'batch_size': batch_size,
            'hidden_dims': [128, 128],
            'kernel': kernel,
            'val_acc': float(acc)

        }
        results.append(result)

    best = sorted(results, key=lambda x: (-x["val_acc"]))[0]

    save_csv("results/svm_results.csv", results)

    with open("results/best_svm_config.json", "w") as f:
        json.dump(best, f, indent=2)

    print("\nBEST FINE ARCH:")
    print(best)
    print("\nSaved:")
    print(" - results/fine_results.csv")
    print(" - results/best_fine_config.json")
    print(" - results/checkpoints/(...).pt")