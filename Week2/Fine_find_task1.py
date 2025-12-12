# Fine_find_task1.py
from typing import *
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms.v2 as F
import tqdm
import itertools
import json
import csv
import os
import wandb


# -------------------------
# Flexible MLP (para variar layers/neurons)
# -------------------------
class FlexibleMLP(nn.Module):
    def __init__(self, input_d: int, hidden_dims: List[int], output_d: int):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_d
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_d))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.net(x)


# -------------------------
# Train / Eval (igual estilo main)
# -------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    correct, total = 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return train_loss / total, correct / total


@torch.no_grad()
def eval_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct, total = 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_transforms(resize: int):
    return F.Compose([
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        F.Resize(size=(resize, resize)),
    ])


def make_loaders(train_dir: str, test_dir: str, resize: int, batch_size: int,
                 val_ratio: float = 0.2, num_workers: int = 8):
    transform = make_transforms(resize)

    dataset_full = ImageFolder(train_dir, transform=transform)
    num_classes = len(dataset_full.classes)

    n_total = len(dataset_full)
    n_val = int(val_ratio * n_total)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset_full, [n_train, n_val], generator=generator)

    test_ds = ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    sample_img, _ = dataset_full[0]
    C, H, W = np.array(sample_img).shape
    input_d = C * H * W

    return train_loader, val_loader, test_loader, input_d, num_classes


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
    set_seed(42)

    # ---- EDITA ESTO ----
    TRAIN_DIR = "/ghome/group03/mcv/datasets/C3/2526/places_reduced/train"
    TEST_DIR  = "/ghome/group03/mcv/datasets/C3/2526/places_reduced/val"
    # --------------------

    WANDB_PROJECT = "C3_DL_fine"
    WANDB_ENTITY = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carga best coarse
    with open("results/best_coarse_config.json", "r") as f:
        best_coarse = json.load(f)

    resize = int(best_coarse["resize"])
    batch_size = int(best_coarse["batch_size"])
    epochs = int(best_coarse["epochs"])   # epochs fijo (entra sí o sí)

    print("Using best coarse:", best_coarse)

    # Loaders fijos
    train_loader, val_loader, test_loader, input_d, num_classes = make_loaders(
        TRAIN_DIR, TEST_DIR, resize=resize, batch_size=batch_size, val_ratio=0.2
    )

    # Grid arquitectura
    widths = [128, 256, 300, 512]
    depths = [1, 2, 3, 5]

    hidden_dim_candidates: List[List[int]] = []
    for d in depths:
        for w in widths:
            hidden_dim_candidates.append([w] * d)

    # “funnel” extras (opcional)
    hidden_dim_candidates += [
        [512, 256],
        [512, 256, 128],
        [300, 200, 100],
    ]

    results = []
    group_name = f"fine_arch_r{resize}_b{batch_size}_e{epochs}"

    for hidden_dims in hidden_dim_candidates:
        cfg = {
            "resize": resize,
            "batch_size": batch_size,
            "epochs": epochs,
            "hidden_dims": hidden_dims,
        }

        model = FlexibleMLP(input_d=input_d, hidden_dims=hidden_dims, output_d=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            group=group_name,
            config=cfg,
            name=f"fine_{'x'.join(map(str, hidden_dims))}",
            reinit=True
        )

        best_val_acc = -1.0
        best_val_loss = 1e9
        best_epoch = -1

        for epoch in tqdm.tqdm(range(epochs), desc=f"FINE hidden={hidden_dims}"):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            va_loss, va_acc = eval_model(model, val_loader, criterion, device)
            te_loss, te_acc = eval_model(model, test_loader, criterion, device)

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                best_val_loss = va_loss
                best_epoch = epoch + 1

            wandb.log({
                "epoch": epoch + 1,
                "train/loss": tr_loss,
                "train/acc": tr_acc,
                "val/loss": va_loss,
                "val/acc": va_acc,
                "test/loss": te_loss,
                "test/acc": te_acc,
                "best/val_acc_so_far": best_val_acc,
            })

        result = {
            "resize": resize,
            "batch_size": batch_size,
            "epochs": epochs,
            "hidden_dims": str(hidden_dims),
            "best_val_acc": float(best_val_acc),
            "best_val_loss": float(best_val_loss),
            "best_epoch": int(best_epoch),
        }
        results.append(result)

        run.summary["best_val_acc"] = best_val_acc
        run.summary["best_epoch"] = best_epoch
        wandb.finish()

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best = sorted(results, key=lambda x: (-x["best_val_acc"], x["best_val_loss"]))[0]

    os.makedirs("results", exist_ok=True)
    save_csv("results/fine_results.csv", results)

    with open("results/best_fine_config.json", "w") as f:
        json.dump(best, f, indent=2)

    print("\nBEST FINE ARCH:")
    print(best)
    print("\nSaved:")
    print(" - results/fine_results.csv")
    print(" - results/best_fine_config.json")
