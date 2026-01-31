# Coarse_find_task1.py
import csv
import itertools
import json
import os
from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as F
import tqdm
import wandb
from dataset import C3Dataset
from models import SimpleModel
from torch.utils.data import DataLoader


def train_one_epoch(model, dataloader: DataLoader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    correct, total = 0, 0

    for inputs, labels in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    # Image log
    img_list = []
    for img, output, label in zip(inputs[-2:], predicted[-2:], labels[-2:]):
        img = img.cpu().detach()
        img = img.permute(1, 2, 0).numpy()
        caption = f"Output: {dataloader.dataset.classes[output.item()]}\nLabel: {dataloader.dataset.classes[label.item()]}"
        img_list.append(wandb.Image(img, caption=caption))

    return train_loss / total, correct / total, img_list


@torch.no_grad()
def eval_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct, total = 0, 0

    for inputs, labels in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    # Image log
    img_list = []
    for img, output, label in zip(inputs[-2:], predicted[-2:], labels[-2:]):
        img = img.cpu().detach()
        img = img.permute(1, 2, 0).numpy()
        caption = f"Output: {dataloader.dataset.classes[output.item()]}\nLabel: {dataloader.dataset.classes[label.item()]}"
        img_list.append(wandb.Image(img, caption=caption))

    return total_loss / total, correct / total, img_list


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_transforms(resize: int):
    train_transforms = F.Compose([
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        F.Resize(size=(resize, resize)),
        F.RandomHorizontalFlip(),
        F.RandomRotation(20)
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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False)

    sample_img, _ = train_ds[0]
    C, H, W = sample_img.shape
    input_d = C * H * W

    num_classes = len(train_ds.classes)

    return train_loader, val_loader, input_d, num_classes


def run_one_config(cfg: Dict, train_dir: str, test_dir: str, device: torch.device,
                   project: str, entity: Optional[str], group: str):
    # loaders
    train_loader, val_loader, input_d, num_classes = make_loaders(
        train_dir=train_dir,
        test_dir=test_dir,
        resize=cfg["resize"],
        batch_size=cfg["batch_size"],
        val_ratio=0.2,
        num_workers=0,
        device=device
    )

    # model baseline FIXED (hidden_d=300, 2 hidden layers)
    model = SimpleModel(input_d=input_d, hidden_d=300, output_d=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # W&B
    run = wandb.init(
        project=project,
        entity=entity,
        group=group,
        config=cfg,
        name=f"coarse_r{cfg['resize']}_b{cfg['batch_size']}",
        reinit=True
    )

    best_val_acc = -1.0
    best_val_loss = 1e9
    best_epoch = -1
    patience = 0

    for epoch in tqdm.tqdm(range(100), desc=f"COARSE r={cfg['resize']} b={cfg['batch_size']}"):
        tr_loss, tr_acc, tr_sample = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, va_sample = eval_model(model, val_loader, criterion, device)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_epoch = epoch + 1

        wandb.log({
            "train/loss": tr_loss,
            "train/acc": tr_acc,
            "val/loss": va_loss,
            "val/acc": va_acc,
            "best/val_acc": best_val_acc,
        }, step=epoch+1)

        if epoch % 5 == 0:
            wandb.log({
                "train/sample": tr_sample,
                "val/sample": va_sample
            }, step=epoch+1)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            patience = 0
        else:
            patience += 1
        
        if patience >= 15:
            break

    result = {
        **cfg,
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(best_epoch),
    }

    run.summary["best_val_acc"] = best_val_acc
    run.summary["best_epoch"] = best_epoch
    wandb.finish()

    # cleanup
    del model, train_loader, val_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def save_csv(path: str, rows: List[Dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


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

    # Grid (epochs entra sí o sí)
    search_space = {
        "resize": [64, 96, 128],
        "batch_size": [64, 128, 256],
    }

    configs = []
    for r, b in itertools.product(search_space["resize"], search_space["batch_size"]):
        configs.append({"resize": r, "batch_size": b})

    results = []
    group_name = "coarse_grid"
    
    for cfg in configs:
        res = run_one_config(cfg, TRAIN_DIR, TEST_DIR, device, WANDB_PROJECT, WANDB_ENTITY, group_name)
        results.append(res)

    # Elige best por best_val_acc (tie-break: menor val_loss)
    best = sorted(results, key=lambda x: (-x["best_val_acc"]))[0]

    os.makedirs("results", exist_ok=True)
    save_csv("results/coarse_results.csv", results)

    with open("results/best_coarse_config.json", "w") as f:
        json.dump(best, f, indent=2)

    print("\nBEST COARSE CONFIG:")
    print(best)
    print("\nSaved:")
    print(" - results/coarse_results.csv")
    print(" - results/best_coarse_config.json")
